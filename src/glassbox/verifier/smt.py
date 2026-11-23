"""Z3 SMT exact formal verifier for piecewise-linear neural networks and IR programs."""

from __future__ import annotations

import fractions
import math
import time
from typing import TYPE_CHECKING, Literal

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
from glassbox.types import (
    Counterexample,
    DecompileConfig,
    RealArithmeticProof,
    TaskType,
    VerificationStatus,
)

if TYPE_CHECKING:
    from glassbox.loader import DenseLayer, LoadedModel

_MAX_BOUND_DEN = 10**9


class SMTResult(BaseModel):
    """Outcome of a Z3 SMT equivalence verification query."""

    checked: bool
    proven: bool
    timed_out: bool
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    counterexample: Counterexample | None = None
    time_ms: float = 0.0
    reason: str = ""
    proof: RealArithmeticProof | None = None


def _r(value: float | fractions.Fraction, ctx: z3.Context | None = None) -> z3.RatNumRef:
    """Convert an IEEE-754 float or Fraction to an exact Z3 rational with zero rounding."""
    if isinstance(value, fractions.Fraction):
        return z3.Q(value.numerator, value.denominator, ctx)
    v = float(value)
    if v == 0.0 or not math.isfinite(v):
        return z3.Q(0, 1, ctx)
    frac = fractions.Fraction(v)
    return z3.Q(frac.numerator, frac.denominator, ctx)


def _outward_lower(q: fractions.Fraction, max_den: int = _MAX_BOUND_DEN) -> fractions.Fraction:
    """Return an exact rational strictly <= q with denominator <= max_den."""
    if q.denominator <= max_den:
        return q
    return fractions.Fraction((q.numerator * max_den) // q.denominator, max_den)


def _outward_upper(q: fractions.Fraction, max_den: int = _MAX_BOUND_DEN) -> fractions.Fraction:
    """Return an exact rational strictly >= q with denominator <= max_den."""
    if q.denominator <= max_den:
        return q
    return fractions.Fraction(-((-q.numerator * max_den) // q.denominator), max_den)


def _encode_expr(expr: Expr, x_vars: list[z3.ArithRef], ctx: z3.Context) -> z3.ArithRef:
    """Recursively encode a piecewise-linear IR expression into exact Z3 rational arithmetic."""
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
    """Encode a LinearGuard hyperplane inequality preserving every nonzero coefficient."""
    terms = [_r(w, ctx) * x_vars[i] for i, w in enumerate(guard.weights) if float(w) != 0.0]
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


def _certified_fraction_layer_bounds(
    layers: list[DenseLayer],
    lower: list[float],
    upper: list[float],
    path_guards: tuple[tuple[LinearGuard, bool], ...],
) -> list[tuple[list[fractions.Fraction], list[fractions.Fraction]]] | None:
    """Compute 100% deductively certified exact Fraction pre-activation bounds per layer.

    Combines exact Fraction Interval Bound Propagation (IBP) with Neumaier-Shcherbina
    exact rational dual LP certification (every numeric LP bound is accepted ONLY when
    certified by exact Fraction dual multipliers y >= 0 over exact rational rows).
    """
    frac_cls = fractions.Fraction
    d = len(lower)
    l_q = [frac_cls(float(v)) for v in lower]
    u_q = [frac_cls(float(v)) for v in upper]

    # Exact Fraction tightening for single-variable (axis-aligned) path guards.
    for guard, is_true in path_guards:
        nz = [(i, frac_cls(float(w))) for i, w in enumerate(guard.weights) if float(w) != 0.0]
        if len(nz) == 1:
            idx, w_q = nz[0]
            b_q = frac_cls(float(guard.bias))
            le = (guard.op in ("<=", "<")) if is_true else (guard.op not in ("<=", "<"))
            bound_q = -b_q / w_q
            if (le and w_q > 0) or ((not le) and w_q < 0):
                if bound_q < u_q[idx]:
                    u_q[idx] = _outward_upper(bound_q)
            else:
                if bound_q > l_q[idx]:
                    l_q[idx] = _outward_lower(bound_q)

    if any(l_q[i] > u_q[i] for i in range(d)):
        return None

    a_q_rows: list[dict[int, fractions.Fraction]] = []
    b_q_rows: list[fractions.Fraction] = []
    a_fl_rows: list[dict[int, float]] = []
    b_fl_rows: list[float] = []

    for guard, is_true in path_guards:
        le = (guard.op in ("<=", "<")) if is_true else (guard.op not in ("<=", "<"))
        sign = 1 if le else -1
        row_q = {
            i: sign * frac_cls(float(w))
            for i, w in enumerate(guard.weights)
            if float(w) != 0.0
        }
        if not row_q:
            continue
        rhs_q = -sign * frac_cls(float(guard.bias))
        a_q_rows.append(row_q)
        b_q_rows.append(rhs_q)
        a_fl_rows.append({i: float(v) for i, v in row_q.items()})
        b_fl_rows.append(float(rhs_q))

    n_vars = d
    prev_off = 0
    total_hidden = sum(int(np.asarray(layer.weight).shape[0]) for layer in layers[:-1])
    use_dual_lp = total_hidden <= 96
    all_bounds: list[tuple[list[fractions.Fraction], list[fractions.Fraction]]] = []

    for li, layer in enumerate(layers):
        w_arr = np.asarray(layer.weight, dtype=np.float64)
        b_arr = np.asarray(layer.bias, dtype=np.float64)
        out_dim, in_dim = w_arr.shape

        if use_dual_lp and a_fl_rows:
            a_mat = np.zeros((len(a_fl_rows), n_vars), dtype=np.float64)
            for ri, rdict in enumerate(a_fl_rows):
                for ci, val in rdict.items():
                    a_mat[ri, ci] = val
            b_vec = np.asarray(b_fl_rows, dtype=np.float64)
            bounds_fl = [(float(l_q[i]), float(u_q[i])) for i in range(n_vars)]
        else:
            a_mat, b_vec, bounds_fl = None, None, None

        pre_l_q: list[fractions.Fraction] = []
        pre_u_q: list[fractions.Fraction] = []

        for j in range(out_dim):
            w_q = [frac_cls(float(w_arr[j, k])) for k in range(in_dim)]
            bias_q = frac_cls(float(b_arr[j]))

            # 1. Exact Fraction Interval Bound Propagation (IBP) baseline.
            ibp_l = bias_q + sum(
                (
                    w_q[k] * l_q[prev_off + k]
                    if w_q[k] >= 0
                    else w_q[k] * u_q[prev_off + k]
                )
                for k in range(in_dim)
                if w_q[k] != 0
            )
            ibp_u = bias_q + sum(
                (
                    w_q[k] * u_q[prev_off + k]
                    if w_q[k] >= 0
                    else w_q[k] * l_q[prev_off + k]
                )
                for k in range(in_dim)
                if w_q[k] != 0
            )

            # 2. Neumaier-Shcherbina exact rational dual LP certificate.
            if a_mat is not None:
                c_fl = np.zeros(n_vars, dtype=np.float64)
                c_fl[prev_off : prev_off + in_dim] = w_arr[j]

                rmin = linprog(c_fl, A_ub=a_mat, b_ub=b_vec, bounds=bounds_fl, method="highs")
                if rmin.success and getattr(rmin, "ineqlin", None) is not None:
                    r_q = [frac_cls(0)] * n_vars
                    for k in range(in_dim):
                        r_q[prev_off + k] = w_q[k]
                    dual_rhs = frac_cls(0)
                    for m_idx, marg in enumerate(rmin.ineqlin.marginals):
                        ym = -float(marg)
                        if ym > 0.0 and math.isfinite(ym):
                            yq = _outward_lower(frac_cls(ym))
                            if yq > 0:
                                dual_rhs += yq * b_q_rows[m_idx]
                                for ci, a_val in a_q_rows[m_idx].items():
                                    r_q[ci] += yq * a_val
                    cert_l = (
                        bias_q
                        - dual_rhs
                        + sum(
                            (r_q[ci] * l_q[ci] if r_q[ci] >= 0 else r_q[ci] * u_q[ci])
                            for ci in range(n_vars)
                            if r_q[ci] != 0
                        )
                    )
                    ibp_l = max(ibp_l, cert_l)

                rmax = linprog(-c_fl, A_ub=a_mat, b_ub=b_vec, bounds=bounds_fl, method="highs")
                if rmax.success and getattr(rmax, "ineqlin", None) is not None:
                    r_q = [frac_cls(0)] * n_vars
                    for k in range(in_dim):
                        r_q[prev_off + k] = w_q[k]
                    dual_rhs = frac_cls(0)
                    for m_idx, marg in enumerate(rmax.ineqlin.marginals):
                        ym = -float(marg)
                        if ym > 0.0 and math.isfinite(ym):
                            yq = _outward_lower(frac_cls(ym))
                            if yq > 0:
                                dual_rhs += yq * b_q_rows[m_idx]
                                for ci, a_val in a_q_rows[m_idx].items():
                                    r_q[ci] -= yq * a_val
                    cert_u = (
                        bias_q
                        + dual_rhs
                        + sum(
                            (r_q[ci] * u_q[ci] if r_q[ci] >= 0 else r_q[ci] * l_q[ci])
                            for ci in range(n_vars)
                            if r_q[ci] != 0
                        )
                    )
                    ibp_u = min(ibp_u, cert_u)

            if ibp_l > ibp_u:
                return None

            pre_l_q.append(_outward_lower(ibp_l))
            pre_u_q.append(_outward_upper(ibp_u))

        all_bounds.append((pre_l_q, pre_u_q))
        if li == len(layers) - 1:
            break

        cur_off = n_vars
        n_vars += out_dim
        for j in range(out_dim):
            w_q_map = {
                prev_off + k: frac_cls(float(w_arr[j, k]))
                for k in range(in_dim)
                if float(w_arr[j, k]) != 0.0
            }
            bias_q = frac_cls(float(b_arr[j]))
            pl = pre_l_q[j]
            pu = pre_u_q[j]
            if layer.activation == "Relu":
                if pu <= 0:
                    l_q.append(frac_cls(0))
                    u_q.append(frac_cls(0))
                elif pl >= 0:
                    l_q.append(pl)
                    u_q.append(pu)
                    r1 = {cur_off + j: frac_cls(1), **{ci: -v for ci, v in w_q_map.items()}}
                    r2 = {cur_off + j: frac_cls(-1), **w_q_map}
                    a_q_rows.extend([r1, r2])
                    b_q_rows.extend([bias_q, -bias_q])
                    a_fl_rows.extend(
                        [
                            {k: float(v) for k, v in r1.items()},
                            {k: float(v) for k, v in r2.items()},
                        ]
                    )
                    b_fl_rows.extend([float(bias_q), float(-bias_q)])
                else:
                    l_q.append(frac_cls(0))
                    u_q.append(pu)
                    r1 = {cur_off + j: frac_cls(-1), **w_q_map}
                    slope_q = _outward_upper(pu / (pu - pl))
                    rhs2 = _outward_upper(slope_q * (bias_q - pl))
                    r2 = {
                        cur_off + j: frac_cls(1),
                        **{ci: -slope_q * v for ci, v in w_q_map.items()},
                    }
                    a_q_rows.extend([r1, r2])
                    b_q_rows.extend([-bias_q, rhs2])
                    a_fl_rows.extend(
                        [
                            {k: float(v) for k, v in r1.items()},
                            {k: float(v) for k, v in r2.items()},
                        ]
                    )
                    b_fl_rows.extend([float(-bias_q), float(rhs2)])
            else:
                l_q.append(pl)
                u_q.append(pu)
                r1 = {cur_off + j: frac_cls(1), **{ci: -v for ci, v in w_q_map.items()}}
                r2 = {cur_off + j: frac_cls(-1), **w_q_map}
                a_q_rows.extend([r1, r2])
                b_q_rows.extend([bias_q, -bias_q])
                a_fl_rows.extend(
                    [
                        {k: float(v) for k, v in r1.items()},
                        {k: float(v) for k, v in r2.items()},
                    ]
                )
                b_fl_rows.extend([float(bias_q), float(-bias_q)])

        prev_off = cur_off

    return all_bounds


def _encode_mlp(
    solver: z3.Solver,
    layers: list[DenseLayer],
    x_vars: list[z3.ArithRef],
    ctx: z3.Context,
    layer_bounds: list[tuple[list[fractions.Fraction], list[fractions.Fraction]]] | None = None,
) -> list[z3.ArithRef]:
    """Encode a ReLU/Identity MLP with exact z3.If and exact rational Planet cuts."""
    h: list[z3.ArithRef] = x_vars
    for layer_idx, layer in enumerate(layers):
        weight = np.asarray(layer.weight, dtype=np.float64)
        bias = np.asarray(layer.bias, dtype=np.float64)
        out_features = weight.shape[0]
        pl, pu = layer_bounds[layer_idx] if layer_bounds is not None else (None, None)
        next_h: list[z3.ArithRef] = []
        for j in range(out_features):
            row = weight[j]
            terms = [_r(w, ctx) * h[k] for k, w in enumerate(row) if float(w) != 0.0]
            pre = (z3.Sum(terms) if terms else _r(0.0, ctx)) + _r(bias[j], ctx)
            var = z3.Real(f"h_{layer_idx}_{j}", ctx)
            if layer.activation == "Relu":
                if pu is not None and pu[j] <= 0:
                    solver.add(var == _r(0.0, ctx))
                elif pl is not None and pl[j] >= 0:
                    solver.add(var == pre)
                else:
                    solver.add(var == z3.If(pre >= 0, pre, _r(0.0, ctx), ctx))
                    if pl is not None and pu is not None:
                        low_q = pl[j]
                        high_q = pu[j]
                        slope_q = _outward_upper(high_q / (high_q - low_q))
                        solver.add(
                            var >= 0,
                            var >= pre,
                            var <= _r(slope_q, ctx) * (pre - _r(low_q, ctx)),
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


def _build_counterexample_from_witness(
    model: LoadedModel,
    program: DecompiledProgram,
    x_star: list[float],
    task_type: TaskType,
    config: DecompileConfig,
    iteration: int,
    z3_model: z3.ModelRef | None = None,
    nn_out_vars: list[z3.ArithRef] | None = None,
    leaf_out_vars: list[z3.ArithRef] | None = None,
    leaf_regime_id: int | None = None,
) -> tuple[Counterexample, bool]:
    """Build a Counterexample from a Z3 witness and report whether float64 confirms it."""
    x_arr = np.asarray([x_star], dtype=np.float64)
    nn_y = np.asarray(model.predict(x_arr)[0], dtype=np.float64)
    prog_y_list, regime_id = program.evaluate_one(x_star)
    prog_y = np.asarray(prog_y_list, dtype=np.float64)
    max_abs_err = float(np.max(np.abs(nn_y - prog_y)))

    if z3_model is not None and nn_out_vars is not None and leaf_out_vars is not None:
        z3_nn = [
            _z3_value_to_float(z3_model.eval(v, model_completion=True), float(nn_y[j]))
            for j, v in enumerate(nn_out_vars)
        ]
        z3_prog = [
            _z3_value_to_float(z3_model.eval(v, model_completion=True), float(prog_y[j]))
            for j, v in enumerate(leaf_out_vars)
        ]
        z3_err = max(abs(a - b) for a, b in zip(z3_nn, z3_prog))
        if z3_err > max_abs_err:
            max_abs_err = float(z3_err)
            prog_y = np.asarray(z3_prog, dtype=np.float64)
            nn_y = np.asarray(z3_nn, dtype=np.float64)
            if leaf_regime_id is not None:
                regime_id = int(leaf_regime_id)

    if task_type == TaskType.REGRESSION:
        confirmed = max_abs_err > config.tolerance_eps
        return (
            Counterexample(
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
            ),
            confirmed,
        )

    nn_cls = (
        int(nn_y[0] > 0.5) if nn_y.shape[0] == 1 else int(np.argmax(nn_y))
    )
    if leaf_regime_id is not None:
        matched = [lf for lf in program.leaves() if lf.regime_id == leaf_regime_id]
        if matched and matched[0].class_id is not None:
            prog_cls = int(matched[0].class_id)
        else:
            prog_cls = (
                int(prog_y[0] > 0.5) if prog_y.shape[0] == 1 else int(np.argmax(prog_y))
            )
    else:
        prog_cls = int(program.predict_classes(x_arr)[0])

    margin = _nn_classification_margin(nn_y, nn_cls)
    confirmed = nn_cls != prog_cls and margin >= max(0.0, config.classification_margin - 1e-7)
    return (
        Counterexample(
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
        ),
        confirmed,
    )


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
) -> tuple[Literal["unsat", "sat", "timeout", "unknown"], Counterexample | None]:
    """Verify a single disjoint leaf polyhedron sequentially in a fresh exact Z3 Context."""
    remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
    if remaining_ms < 50:
        return "timeout", None

    domain = model.metadata.domain
    is_classification = task_type == TaskType.CLASSIFICATION
    output_dim = domain.output_dim

    layer_bounds = _certified_fraction_layer_bounds(
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

    # If certified bounds reported infeasible leaf, confirm exact infeasibility in Z3.
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
        if float(config.classification_margin) <= 0.0:
            nn_cls_expr = _encode_argmax(nn_out, ctx)
            violation = nn_cls_expr != leaf_cls
        else:
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

    t_check = time.perf_counter()
    status = solver.check()
    check_ms = (time.perf_counter() - t_check) * 1000.0
    if status == z3.unsat:
        return "unsat", None
    if status == z3.unknown:
        if time.perf_counter() >= deadline - 0.05 or check_ms >= remaining_ms * 0.8:
            return "timeout", None
        return "unknown", None

    # Exact rational Z3 returned SAT: a true real-arithmetic counterexample exists.
    z3_m = solver.model()
    x_star = _extract_model_input(z3_m, x_vars, domain.lower_bounds, domain.upper_bounds)
    first_cex, confirmed = _build_counterexample_from_witness(
        model,
        program,
        x_star,
        task_type,
        config,
        iteration,
        z3_model=z3_m,
        nn_out_vars=nn_out,
        leaf_out_vars=leaf_outs,
        leaf_regime_id=leaf.regime_id,
    )
    if confirmed:
        return "sat", first_cex

    # If the first float projection landed on a boundary, try nudging inside the SAT region.
    spans = [hi - lo for lo, hi in zip(domain.lower_bounds, domain.upper_bounds)]
    for _ in range(2):
        remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
        if remaining_ms < 50:
            break
        solver.set("timeout", remaining_ms)
        exclude = z3.Or(
            [
                z3.Or(
                    x_vars[i] < _r(x_star[i] - 1e-5 * spans[i], ctx),
                    x_vars[i] > _r(x_star[i] + 1e-5 * spans[i], ctx),
                    ctx,
                )
                for i in range(domain.input_dim)
            ],
            ctx,
        )
        solver.add(exclude)
        if solver.check() != z3.sat:
            break
        z3_m = solver.model()
        x_star = _extract_model_input(z3_m, x_vars, domain.lower_bounds, domain.upper_bounds)
        cand_cex, confirmed = _build_counterexample_from_witness(
            model,
            program,
            x_star,
            task_type,
            config,
            iteration,
            z3_model=z3_m,
            nn_out_vars=nn_out,
            leaf_out_vars=leaf_outs,
            leaf_regime_id=leaf.regime_id,
        )
        if confirmed:
            return "sat", cand_cex

    # Never suppress an exact rational SAT witness.
    return "sat", first_cex


def _skipped_result(
    program: DecompiledProgram,
    config: DecompileConfig,
    reason: str,
    t0: float,
) -> SMTResult:
    leaves = program.leaves()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    proof = RealArithmeticProof(
        exact_rationals=True,
        sound_bounds_only=True,
        scope="none",
        tolerance_eps=float(config.tolerance_eps),
        classification_margin=float(config.classification_margin),
        z3_version=z3.get_version_string(),
        solver_result="skipped",
        leaves_proven=0,
        leaves_total=len(leaves),
        leaf_proven_map={lf.regime_id: False for lf in leaves},
        solve_time_ms=elapsed_ms,
        notes=reason,
    )
    return SMTResult(
        checked=False,
        proven=False,
        timed_out=False,
        status=VerificationStatus.UNVERIFIED,
        counterexample=None,
        time_ms=elapsed_ms,
        reason=reason,
        proof=proof,
    )


def verify_smt(
    model: LoadedModel,
    program: DecompiledProgram,
    config: DecompileConfig,
    iteration: int = 0,
) -> SMTResult:
    """Formally verify equivalence of a piecewise-linear MLP and IR program using exact Z3 QF_LRA."""
    t0 = time.perf_counter()

    if not config.verify_smt:
        return _skipped_result(program, config, "smt_disabled", t0)
    if not model.metadata.is_piecewise_linear:
        return _skipped_result(program, config, "model_not_piecewise_linear", t0)
    if not program.is_piecewise_linear():
        return _skipped_result(program, config, "program_not_piecewise_linear", t0)

    layers = model.extract_mlp_layers()
    if layers is None or not layers:
        return _skipped_result(program, config, "non_mlp_architecture", t0)
    if any(layer.activation not in {"Relu", "Identity"} for layer in layers):
        return _skipped_result(program, config, "unsupported_activation", t0)

    domain = model.metadata.domain
    task_type = config.task_type or domain.task_type
    is_classification = task_type == TaskType.CLASSIFICATION
    scope: Literal["global", "margin_conditioned"] = (
        "margin_conditioned"
        if (is_classification and float(config.classification_margin) > 0.0)
        else "global"
    )
    proven_status = (
        VerificationStatus.PROVEN_REAL_MARGIN
        if scope == "margin_conditioned"
        else VerificationStatus.PROVEN_REAL_GLOBAL
    )

    deadline = t0 + max(0.1, float(config.smt_timeout_s))
    leaf_paths = _collect_leaf_paths(program.root)
    leaf_paths.sort(key=lambda item: float(item[1].local_error), reverse=True)
    leaf_proven_map: dict[int, bool] = {lf.regime_id: False for _, lf in leaf_paths}

    outcomes: list[tuple[str, Counterexample | None]] = []
    for path_guards, leaf in leaf_paths:
        kind, cex = _verify_single_leaf(
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
        outcomes.append((kind, cex))
        if kind == "unsat":
            leaf_proven_map[leaf.regime_id] = True
        elif kind == "sat" and cex is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            proof = RealArithmeticProof(
                exact_rationals=True,
                sound_bounds_only=True,
                scope=scope,
                tolerance_eps=float(config.tolerance_eps),
                classification_margin=float(config.classification_margin),
                z3_version=z3.get_version_string(),
                solver_result="sat",
                leaves_proven=sum(1 for v in leaf_proven_map.values() if v),
                leaves_total=len(leaf_paths),
                leaf_proven_map=leaf_proven_map,
                solve_time_ms=elapsed_ms,
                notes=f"Counterexample found in regime {leaf.regime_id}.",
            )
            return SMTResult(
                checked=True,
                proven=False,
                timed_out=False,
                status=VerificationStatus.COUNTEREXAMPLE_FOUND,
                counterexample=cex,
                time_ms=elapsed_ms,
                reason="counterexample_found",
                proof=proof,
            )
        elif kind == "timeout" and time.perf_counter() >= deadline:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    leaves_proven = sum(1 for v in leaf_proven_map.values() if v)

    if len(outcomes) == len(leaf_paths) and outcomes and all(k == "unsat" for k, _ in outcomes):
        proof = RealArithmeticProof(
            exact_rationals=True,
            sound_bounds_only=True,
            scope=scope,
            tolerance_eps=float(config.tolerance_eps),
            classification_margin=float(config.classification_margin),
            z3_version=z3.get_version_string(),
            solver_result="unsat",
            leaves_proven=leaves_proven,
            leaves_total=len(leaf_paths),
            leaf_proven_map=leaf_proven_map,
            solve_time_ms=elapsed_ms,
            notes="All leaves proven unsat over exact rational QF_LRA.",
        )
        return SMTResult(
            checked=True,
            proven=True,
            timed_out=False,
            status=proven_status,
            counterexample=None,
            time_ms=elapsed_ms,
            reason="unsat_proven",
            proof=proof,
        )

    if len(outcomes) < len(leaf_paths) or any(k == "timeout" for k, _ in outcomes):
        proof = RealArithmeticProof(
            exact_rationals=True,
            sound_bounds_only=True,
            scope=scope,
            tolerance_eps=float(config.tolerance_eps),
            classification_margin=float(config.classification_margin),
            z3_version=z3.get_version_string(),
            solver_result="timeout",
            leaves_proven=leaves_proven,
            leaves_total=len(leaf_paths),
            leaf_proven_map=leaf_proven_map,
            solve_time_ms=elapsed_ms,
            notes=f"Timed out after proving {leaves_proven}/{len(leaf_paths)} leaves.",
        )
        return SMTResult(
            checked=True,
            proven=False,
            timed_out=True,
            status=VerificationStatus.TIMEOUT,
            counterexample=None,
            time_ms=elapsed_ms,
            reason="z3_timeout",
            proof=proof,
        )

    proof = RealArithmeticProof(
        exact_rationals=True,
        sound_bounds_only=True,
        scope=scope,
        tolerance_eps=float(config.tolerance_eps),
        classification_margin=float(config.classification_margin),
        z3_version=z3.get_version_string(),
        solver_result="unknown",
        leaves_proven=leaves_proven,
        leaves_total=len(leaf_paths),
        leaf_proven_map=leaf_proven_map,
        solve_time_ms=elapsed_ms,
        notes="Z3 returned unknown without timeout.",
    )
    return SMTResult(
        checked=True,
        proven=False,
        timed_out=False,
        status=VerificationStatus.UNKNOWN,
        counterexample=None,
        time_ms=elapsed_ms,
        reason="z3_unknown",
        proof=proof,
    )


__all__ = ["SMTResult", "verify_smt"]
