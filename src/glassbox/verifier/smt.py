"""Z3 SMT exact formal verifier for piecewise-linear neural networks and IR programs."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
import z3
from pydantic import BaseModel
from scipy.optimize import linprog

from glassbox.ir import (
    BinaryExpr,
    ClampExpr,
    ConstExpr,
    DecompiledProgram,
    Expr,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    RegimeNode,
    UnaryExpr,
    VarExpr,
)
from glassbox.types import Counterexample, DecompileConfig, TaskType

if TYPE_CHECKING:
    from glassbox.loader import DenseLayer, LoadedModel


class SMTResult(BaseModel):
    """Outcome of a Z3 SMT equivalence verification query."""

    checked: bool
    proven: bool
    timed_out: bool
    counterexample: Counterexample | None = None
    time_ms: float = 0.0
    reason: str = ""


def _r(value: float, ctx: z3.Context | None = None) -> z3.RatNumRef:
    """Convert a float to a compact Z3 rational to avoid decimal denominator blowup."""
    v = float(value)
    if v == 0.0 or not math.isfinite(v):
        return z3.RealVal("0", ctx)
    return z3.RealVal(f"{v:.8g}", ctx)


def _encode_expr(expr: Expr, x_vars: list[z3.ArithRef], ctx: z3.Context) -> z3.ArithRef:
    """Recursively encode a piecewise-linear IR expression into Z3 real arithmetic."""
    if isinstance(expr, VarExpr):
        return x_vars[expr.index]
    if isinstance(expr, ConstExpr):
        return _r(expr.value, ctx)
    if isinstance(expr, UnaryExpr):
        operand = _encode_expr(expr.operand, x_vars, ctx)
        if expr.op == "neg":
            return -operand
        if expr.op == "abs":
            return z3.If(operand >= 0, operand, -operand, ctx)
        if expr.op == "relu":
            return z3.If(operand >= 0, operand, _r(0.0, ctx), ctx)
        raise ValueError(f"Non-linear unary op cannot be encoded in QF_LRA: {expr.op}")
    if isinstance(expr, BinaryExpr):
        left = _encode_expr(expr.left, x_vars, ctx)
        right = _encode_expr(expr.right, x_vars, ctx)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "div":
            return left / right
        if expr.op == "min":
            return z3.If(left <= right, left, right, ctx)
        if expr.op == "max":
            return z3.If(left >= right, left, right, ctx)
        raise ValueError(f"Unsupported binary op: {expr.op}")
    if isinstance(expr, ClampExpr):
        operand = _encode_expr(expr.operand, x_vars, ctx)
        low = _r(expr.low, ctx)
        high = _r(expr.high, ctx)
        return z3.If(operand < low, low, z3.If(operand > high, high, operand, ctx), ctx)
    raise TypeError(f"Unsupported expression type: {type(expr)!r}")


def _encode_guard(guard: LinearGuard, x_vars: list[z3.ArithRef], ctx: z3.Context) -> z3.BoolRef:
    """Encode a LinearGuard hyperplane inequality into a Z3 boolean expression."""
    terms = [_r(w, ctx) * x_vars[i] for i, w in enumerate(guard.weights) if abs(float(w)) > 1e-12]
    lhs = (z3.Sum(terms) if terms else _r(0.0, ctx)) + _r(guard.bias, ctx)
    if guard.op == "<=":
        return lhs <= 0
    if guard.op == "<":
        return lhs < 0
    if guard.op == ">=":
        return lhs >= 0
    if guard.op == ">":
        return lhs > 0
    raise ValueError(f"Unsupported guard op: {guard.op}")


def _encode_argmax(logits: list[z3.ArithRef], ctx: z3.Context) -> z3.ArithRef:
    """Encode deterministic first-index-wins argmax matching numpy.argmax."""
    if len(logits) == 1:
        return z3.If(logits[0] > _r(0.5, ctx), z3.IntVal(1, ctx), z3.IntVal(0, ctx), ctx)
    best_val = logits[0]
    best_idx: z3.ArithRef = z3.IntVal(0, ctx)
    for k in range(1, len(logits)):
        take_k = logits[k] > best_val
        best_idx = z3.If(take_k, z3.IntVal(k, ctx), best_idx, ctx)
        best_val = z3.If(take_k, logits[k], best_val, ctx)
    return best_idx


def _collect_leaf_paths(
    node: RegimeNode,
    path: tuple[tuple[LinearGuard, bool], ...] = (),
) -> list[tuple[tuple[tuple[LinearGuard, bool], ...], RegimeLeaf]]:
    """Decompose a GuardNode decision tree into exhaustive, disjoint leaf polyhedra."""
    if isinstance(node, RegimeLeaf):
        return [(path, node)]
    assert isinstance(node, GuardNode)
    return _collect_leaf_paths(node.true_branch, (*path, (node.guard, True))) + _collect_leaf_paths(
        node.false_branch, (*path, (node.guard, False))
    )


def _compute_leaf_lp_bounds(
    layers: list[DenseLayer],
    lower: list[float],
    upper: list[float],
    path_guards: tuple[tuple[LinearGuard, bool], ...],
    pad: float = 1e-3,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Compute sound outer pre-activation bounds per layer on a leaf polyhedron via Planet LP."""
    d = len(lower)
    total_neurons = sum(int(np.asarray(layer.weight).shape[0]) for layer in layers[:-1])
    if total_neurons > 96:
        l_vec = np.asarray(lower, dtype=np.float64)
        u_vec = np.asarray(upper, dtype=np.float64)
        ibp: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in layers:
            w = np.asarray(layer.weight, dtype=np.float64)
            b = np.asarray(layer.bias, dtype=np.float64)
            wp = np.maximum(w, 0.0)
            wn = np.minimum(w, 0.0)
            pre_l = wp @ l_vec + wn @ u_vec + b - pad
            pre_u = wp @ u_vec + wn @ l_vec + b + pad
            ibp.append((pre_l, pre_u))
            if layer.activation == "Relu":
                l_vec = np.maximum(0.0, pre_l)
                u_vec = np.maximum(0.0, pre_u)
            else:
                l_vec, u_vec = pre_l, pre_u
        return ibp

    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    bounds: list[tuple[float, float]] = list(zip(lower, upper))
    n_vars = d

    for guard, is_true in path_guards:
        w = np.asarray(guard.weights, dtype=np.float64)
        b = float(guard.bias)
        le = guard.op in ("<=", "<")
        if not is_true:
            le = not le
        a_rows.append(w if le else -w)
        b_rows.append(-b if le else b)

    def _build_ab(cols: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not a_rows:
            return None, None
        mat = np.zeros((len(a_rows), cols), dtype=np.float64)
        for idx, row in enumerate(a_rows):
            mat[idx, : len(row)] = row
        return mat, np.asarray(b_rows, dtype=np.float64)

    all_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    prev_off = 0
    prev_dim = d

    for li, layer in enumerate(layers):
        w = np.asarray(layer.weight, dtype=np.float64)
        bias = np.asarray(layer.bias, dtype=np.float64)
        out_dim = w.shape[0]
        a_mat, b_vec = _build_ab(n_vars)

        pre_l = np.empty(out_dim, dtype=np.float64)
        pre_u = np.empty(out_dim, dtype=np.float64)
        for j in range(out_dim):
            c = np.zeros(n_vars, dtype=np.float64)
            c[prev_off : prev_off + prev_dim] = w[j]
            rmin = linprog(c, A_ub=a_mat, b_ub=b_vec, bounds=bounds, method="highs")
            if not rmin.success or rmin.fun is None:
                return None
            rmax = linprog(-c, A_ub=a_mat, b_ub=b_vec, bounds=bounds, method="highs")
            if not rmax.success or rmax.fun is None:
                return None
            pre_l[j] = float(rmin.fun) + float(bias[j]) - pad
            pre_u[j] = float(-rmax.fun) + float(bias[j]) + pad

        all_bounds.append((pre_l, pre_u))
        if li == len(layers) - 1:
            break

        cur_off = n_vars
        n_vars += out_dim
        for j in range(out_dim):
            if layer.activation == "Relu":
                if pre_u[j] <= 0.0:
                    bounds.append((0.0, 0.0))
                elif pre_l[j] >= 0.0:
                    bounds.append((max(0.0, float(pre_l[j])), float(pre_u[j])))
                    r1 = np.zeros(n_vars, dtype=np.float64)
                    r1[cur_off + j] = 1.0
                    r1[prev_off : prev_off + prev_dim] = -w[j]
                    a_rows.append(r1)
                    b_rows.append(float(bias[j]))
                    r2 = np.zeros(n_vars, dtype=np.float64)
                    r2[cur_off + j] = -1.0
                    r2[prev_off : prev_off + prev_dim] = w[j]
                    a_rows.append(r2)
                    b_rows.append(-float(bias[j]))
                else:
                    bounds.append((0.0, float(pre_u[j])))
                    r1 = np.zeros(n_vars, dtype=np.float64)
                    r1[cur_off + j] = -1.0
                    r1[prev_off : prev_off + prev_dim] = w[j]
                    a_rows.append(r1)
                    b_rows.append(-float(bias[j]))
                    slope = float(pre_u[j] / (pre_u[j] - pre_l[j]))
                    r2 = np.zeros(n_vars, dtype=np.float64)
                    r2[cur_off + j] = 1.0
                    r2[prev_off : prev_off + prev_dim] = -slope * w[j]
                    a_rows.append(r2)
                    b_rows.append(slope * (float(bias[j]) - float(pre_l[j])))
            else:
                bounds.append((float(pre_l[j]), float(pre_u[j])))
                r1 = np.zeros(n_vars, dtype=np.float64)
                r1[cur_off + j] = 1.0
                r1[prev_off : prev_off + prev_dim] = -w[j]
                a_rows.append(r1)
                b_rows.append(float(bias[j]))
                r2 = np.zeros(n_vars, dtype=np.float64)
                r2[cur_off + j] = -1.0
                r2[prev_off : prev_off + prev_dim] = w[j]
                a_rows.append(r2)
                b_rows.append(-float(bias[j]))

        prev_off = cur_off
        prev_dim = out_dim

    return all_bounds


def _encode_mlp(
    solver: z3.Solver,
    layers: list[DenseLayer],
    x_vars: list[z3.ArithRef],
    ctx: z3.Context,
    layer_bounds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[z3.ArithRef]:
    """Encode a ReLU/Identity MLP with exact z3.If and optional sound Planet LP cuts."""
    h: list[z3.ArithRef] = x_vars
    for layer_idx, layer in enumerate(layers):
        weight = np.asarray(layer.weight, dtype=np.float64)
        bias = np.asarray(layer.bias, dtype=np.float64)
        out_features = weight.shape[0]
        pl, pu = layer_bounds[layer_idx] if layer_bounds is not None else (None, None)
        next_h: list[z3.ArithRef] = []
        for j in range(out_features):
            row = weight[j]
            terms = [_r(w, ctx) * h[k] for k, w in enumerate(row) if abs(float(w)) > 1e-12]
            pre = (z3.Sum(terms) if terms else _r(0.0, ctx)) + _r(bias[j], ctx)
            var = z3.Real(f"h_{layer_idx}_{j}", ctx)
            if layer.activation == "Relu":
                if pu is not None and float(pu[j]) <= 0.0:
                    solver.add(var == _r(0.0, ctx))
                elif pl is not None and float(pl[j]) >= 0.0:
                    solver.add(var == pre)
                else:
                    solver.add(var == z3.If(pre >= 0, pre, _r(0.0, ctx), ctx))
                    if pl is not None and pu is not None:
                        low_j = float(pl[j])
                        high_j = float(pu[j])
                        slope = high_j / (high_j - low_j)
                        solver.add(
                            var >= 0,
                            var >= pre,
                            var <= _r(slope, ctx) * (pre - _r(low_j, ctx)),
                        )
            elif layer.activation == "Identity":
                solver.add(var == pre)
            else:
                raise ValueError(f"Unsupported SMT layer activation: {layer.activation}")
            next_h.append(var)
        h = next_h
    return h


def _z3_value_to_float(val: z3.ExprRef | None, default: float) -> float:
    """Convert a Z3 model evaluation value into a Python float."""
    if val is None:
        return default
    if z3.is_rational_value(val):
        return float(val.as_fraction())
    if z3.is_int_value(val):
        return float(val.as_long())
    if z3.is_algebraic_value(val):
        return float(val.approx(20).as_fraction())
    try:
        return float(val.as_decimal(16).rstrip("?"))
    except (AttributeError, ValueError, z3.Z3Exception):
        return default


def _extract_model_input(
    z3_model: z3.ModelRef,
    x_vars: list[z3.ArithRef],
    lower: list[float],
    upper: list[float],
) -> list[float]:
    """Extract a concrete bounded float vector from a satisfying Z3 model."""
    coords: list[float] = []
    for var, lo, hi in zip(x_vars, lower, upper):
        raw = z3_model.eval(var, model_completion=True)
        val = _z3_value_to_float(raw, 0.5 * (lo + hi))
        coords.append(float(min(hi, max(lo, val))))
    return coords


def _nn_classification_margin(nn_y: np.ndarray, nn_cls: int) -> float:
    """Compute top-1 vs top-2 margin (or distance from 0.5 for 1D output)."""
    if nn_y.shape[0] == 1:
        return float(abs(nn_y[0] - 0.5))
    others = [float(nn_y[k]) for k in range(nn_y.shape[0]) if k != nn_cls]
    return float(nn_y[nn_cls] - max(others))


def _check_real_violation(
    model: LoadedModel,
    program: DecompiledProgram,
    x_star: list[float],
    task_type: TaskType,
    config: DecompileConfig,
    iteration: int,
) -> Counterexample | None:
    """Verify a Z3 witness against real ONNXRuntime and IR evaluation."""
    x_arr = np.asarray([x_star], dtype=np.float64)
    nn_y = np.asarray(model.predict(x_arr)[0], dtype=np.float64)
    prog_y_list, regime_id = program.evaluate_one(x_star)
    prog_y = np.asarray(prog_y_list, dtype=np.float64)
    max_abs_err = float(np.max(np.abs(nn_y - prog_y)))

    if task_type == TaskType.REGRESSION:
        if max_abs_err > config.tolerance_eps:
            return Counterexample(
                iteration=iteration,
                source="z3_smt",
                x=x_star,
                nn_output=nn_y.tolist(),
                program_output=prog_y.tolist(),
                nn_class=None,
                program_class=None,
                max_abs_error=max_abs_err,
                regime_id=int(regime_id),
                repaired=False,
            )
        return None

    nn_cls = int(model.predict_classes(x_arr)[0])
    prog_cls = int(program.predict_classes(x_arr)[0])
    margin = _nn_classification_margin(nn_y, nn_cls)
    if nn_cls != prog_cls and margin >= config.classification_margin - 1e-6:
        return Counterexample(
            iteration=iteration,
            source="z3_smt",
            x=x_star,
            nn_output=nn_y.tolist(),
            program_output=prog_y.tolist(),
            nn_class=nn_cls,
            program_class=prog_cls,
            max_abs_error=max_abs_err,
            regime_id=int(regime_id),
            repaired=False,
        )
    return None


def _verify_single_leaf(
    model: LoadedModel,
    program: DecompiledProgram,
    layers: list[DenseLayer],
    path_guards: tuple[tuple[LinearGuard, bool], ...],
    leaf: RegimeLeaf,
    task_type: TaskType,
    config: DecompileConfig,
    iteration: int,
    deadline: float,
) -> tuple[str, Counterexample | None]:
    """Verify a single disjoint leaf polyhedron sequentially in a fresh Z3 Context."""
    remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
    if remaining_ms < 50:
        return "timeout", None

    domain = model.metadata.domain
    is_classification = task_type == TaskType.CLASSIFICATION
    output_dim = domain.output_dim

    # HiGHS LP runs strictly on the caller thread to avoid cross-thread C++ pool deadlocks.
    layer_bounds = _compute_leaf_lp_bounds(
        layers, domain.lower_bounds, domain.upper_bounds, path_guards
    )

    remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
    if remaining_ms < 50:
        return "timeout", None

    ctx = z3.Context()
    solver = z3.Then(
        z3.Tactic("simplify", ctx),
        z3.Tactic("solve-eqs", ctx),
        z3.Tactic("smt", ctx),
    ).solver()
    solver.set("timeout", max(50, remaining_ms))

    x_vars = [z3.Real(f"x_{i}", ctx) for i in range(domain.input_dim)]
    for i, (lo, hi) in enumerate(zip(domain.lower_bounds, domain.upper_bounds)):
        solver.add(x_vars[i] >= _r(lo, ctx), x_vars[i] <= _r(hi, ctx))
    for guard, is_true in path_guards:
        cond = _encode_guard(guard, x_vars, ctx)
        solver.add(cond if is_true else z3.Not(cond, ctx))

    # If LP reported infeasible, confirm exact infeasibility in Z3 before returning unsat.
    if layer_bounds is None:
        feas = solver.check()
        if feas == z3.unsat:
            return "unsat", None
        if feas == z3.unknown:
            return "timeout", None

    nn_out = _encode_mlp(solver, layers, x_vars, ctx, layer_bounds=layer_bounds)
    leaf_outs = [_encode_expr(e, x_vars, ctx) for e in leaf.output_exprs]

    if not is_classification:
        eps = _r(config.tolerance_eps, ctx)
        violation = z3.Or(
            [
                z3.Or(nn_out[j] - leaf_outs[j] > eps, leaf_outs[j] - nn_out[j] > eps, ctx)
                for j in range(output_dim)
            ],
            ctx,
        )
    else:
        leaf_cls = (
            z3.IntVal(int(leaf.class_id), ctx)
            if leaf.class_id is not None
            else _encode_argmax(leaf_outs, ctx)
        )
        margin = _r(config.classification_margin, ctx)
        if output_dim >= 2:
            disagreements = []
            for c1 in range(output_dim):
                nn_wins_c1 = z3.And(
                    [
                        nn_out[c1] - nn_out[c2] >= margin
                        for c2 in range(output_dim)
                        if c2 != c1
                    ],
                    ctx,
                )
                disagreements.append(z3.And(nn_wins_c1, leaf_cls != z3.IntVal(c1, ctx), ctx))
            violation = z3.Or(disagreements, ctx)
        else:
            half = _r(0.5, ctx)
            violation = z3.Or(
                z3.And(nn_out[0] - half >= margin, leaf_cls != z3.IntVal(1, ctx), ctx),
                z3.And(half - nn_out[0] >= margin, leaf_cls != z3.IntVal(0, ctx), ctx),
                ctx,
            )

    solver.add(violation)

    remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
    if remaining_ms < 50:
        return "timeout", None
    solver.set("timeout", max(50, remaining_ms))

    status = solver.check()
    if status == z3.unsat:
        return "unsat", None
    if status == z3.unknown:
        return "timeout", None

    spans = [hi - lo for lo, hi in zip(domain.lower_bounds, domain.upper_bounds)]
    for attempt in range(4):
        x_star = _extract_model_input(
            solver.model(), x_vars, domain.lower_bounds, domain.upper_bounds
        )
        cex = _check_real_violation(model, program, x_star, task_type, config, iteration)
        if cex is not None:
            return "sat", cex
        if attempt == 3:
            break
        remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
        if remaining_ms < 50:
            return "timeout", None
        solver.set("timeout", remaining_ms)
        exclude = z3.Or(
            [
                z3.Or(
                    x_vars[i] < _r(x_star[i] - 1e-4 * spans[i], ctx),
                    x_vars[i] > _r(x_star[i] + 1e-4 * spans[i], ctx),
                    ctx,
                )
                for i in range(domain.input_dim)
            ],
            ctx,
        )
        solver.add(exclude)
        next_status = solver.check()
        if next_status != z3.sat:
            return ("timeout" if next_status == z3.unknown else "spurious"), None

    return "spurious", None


def verify_smt(
    model: LoadedModel,
    program: DecompiledProgram,
    config: DecompileConfig,
    iteration: int = 0,
) -> SMTResult:
    """Formally verify equivalence of a piecewise-linear MLP and IR program using Z3."""
    t0 = time.perf_counter()

    if not config.verify_smt:
        return SMTResult(checked=False, proven=False, timed_out=False, reason="smt_disabled")
    if not model.metadata.is_piecewise_linear:
        return SMTResult(
            checked=False,
            proven=False,
            timed_out=False,
            reason="model_not_piecewise_linear",
        )
    if not program.is_piecewise_linear():
        return SMTResult(
            checked=False,
            proven=False,
            timed_out=False,
            reason="program_not_piecewise_linear",
        )

    layers = model.extract_mlp_layers()
    if layers is None or not layers:
        return SMTResult(
            checked=False,
            proven=False,
            timed_out=False,
            reason="non_mlp_architecture",
        )
    if any(layer.activation not in {"Relu", "Identity"} for layer in layers):
        return SMTResult(
            checked=False,
            proven=False,
            timed_out=False,
            reason="unsupported_activation",
        )

    domain = model.metadata.domain
    task_type = config.task_type or domain.task_type
    deadline = t0 + max(0.1, float(config.smt_timeout_s))
    leaf_paths = _collect_leaf_paths(program.root)
    # Check highest-local-error leaves first to surface counterexamples rapidly.
    leaf_paths.sort(key=lambda item: float(item[1].local_error), reverse=True)

    outcomes: list[tuple[str, Counterexample | None]] = []
    for path_guards, leaf in leaf_paths:
        outcome = _verify_single_leaf(
            model,
            program,
            layers,
            path_guards,
            leaf,
            task_type,
            config,
            iteration,
            deadline,
        )
        outcomes.append(outcome)
        if outcome[0] == "sat" and outcome[1] is not None:
            return SMTResult(
                checked=True,
                proven=False,
                timed_out=False,
                counterexample=outcome[1],
                time_ms=(time.perf_counter() - t0) * 1000.0,
                reason="counterexample_found",
            )
        if outcome[0] == "timeout" and time.perf_counter() >= deadline:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if len(outcomes) == len(leaf_paths) and outcomes and all(k == "unsat" for k, _ in outcomes):
        return SMTResult(
            checked=True,
            proven=True,
            timed_out=False,
            counterexample=None,
            time_ms=elapsed_ms,
            reason="unsat_proven",
        )

    if len(outcomes) < len(leaf_paths) or any(k == "timeout" for k, _ in outcomes):
        return SMTResult(
            checked=True,
            proven=False,
            timed_out=True,
            counterexample=None,
            time_ms=elapsed_ms,
            reason="z3_timeout",
        )

    return SMTResult(
        checked=True,
        proven=False,
        timed_out=False,
        counterexample=None,
        time_ms=elapsed_ms,
        reason="spurious_rounding_boundary",
    )


__all__ = ["SMTResult", "verify_smt"]
