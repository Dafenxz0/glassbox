"""Symbolic expression synthesis over extracted regime partitions."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from sklearn.linear_model import OrthogonalMatchingPursuit, Ridge
from sklearn.preprocessing import PolynomialFeatures

from glassbox.ir import (
    BinaryExpr,
    ConstExpr,
    DecompiledProgram,
    Expr,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    RegimeNode,
    UnaryExpr,
    VarExpr,
    eval_expr,
)
from glassbox.types import DecompileConfig, TaskType

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel
    from glassbox.regimes import RegimePartition

_OP_NEGATION = {
    "<=": ">",
    "<": ">=",
    ">=": "<",
    ">": "<=",
}


def linear_to_expr(
    coeffs: Sequence[float] | np.ndarray,
    bias: float,
    feature_names: Sequence[str],
    zero_tol: float = 1e-4,
) -> Expr:
    """Build a compact IR Expr from a linear combination w . x + b."""
    if len(coeffs) != len(feature_names):
        raise ValueError(
            f"coeffs length ({len(coeffs)}) != feature_names length ({len(feature_names)})"
        )

    acc: Expr | None = None
    for idx, (raw_c, fname) in enumerate(zip(coeffs, feature_names)):
        c = float(raw_c)
        if abs(c) <= zero_tol:
            continue
        var = VarExpr(index=idx, name=fname)
        if acc is None:
            if abs(c - 1.0) <= 1e-9:
                acc = var
            elif abs(c + 1.0) <= 1e-9:
                acc = UnaryExpr(op="neg", operand=var)
            else:
                acc = BinaryExpr(op="mul", left=ConstExpr(value=c), right=var)
        elif c < 0.0:
            rhs: Expr = (
                var
                if abs(c + 1.0) <= 1e-9
                else BinaryExpr(op="mul", left=ConstExpr(value=-c), right=var)
            )
            acc = BinaryExpr(op="sub", left=acc, right=rhs)
        else:
            rhs = (
                var
                if abs(c - 1.0) <= 1e-9
                else BinaryExpr(op="mul", left=ConstExpr(value=c), right=var)
            )
            acc = BinaryExpr(op="add", left=acc, right=rhs)

    b = float(bias)
    if acc is None:
        return ConstExpr(value=0.0 if abs(b) <= zero_tol else b)
    if abs(b) > zero_tol:
        if b > 0.0:
            acc = BinaryExpr(op="add", left=acc, right=ConstExpr(value=b))
        else:
            acc = BinaryExpr(op="sub", left=acc, right=ConstExpr(value=-b))
    return acc


def _pow_int(base: Expr, exp: int) -> Expr:
    if exp == 1:
        return base
    if exp == 2:
        return BinaryExpr(op="mul", left=base, right=base)
    half = _pow_int(base, exp // 2)
    sq = BinaryExpr(op="mul", left=half, right=half)
    if exp % 2 == 0:
        return sq
    return BinaryExpr(op="mul", left=sq, right=base)


def _split_sympy_sign(term: sp.Basic) -> tuple[bool, sp.Basic]:
    if isinstance(term, sp.Number):
        val = float(term.evalf())
        return (val < 0.0, sp.Float(abs(val)))
    if isinstance(term, sp.Mul):
        coeff, rest = term.as_coeff_Mul()
        val = float(coeff.evalf())
        if val < 0.0:
            return (True, sp.Mul(-coeff, rest))
    return (False, term)


def sympy_to_expr(
    sym_expr: sp.Expr,
    feature_names: Sequence[str],
    zero_tol: float = 1e-6,
) -> Expr:
    """Convert a SymPy expression into a compact Glassbox IR Expr, pruning near-zero constants."""
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    def _convert(node: sp.Basic) -> Expr:
        if node in (sp.nan, sp.oo, -sp.oo, sp.zoo):
            raise ValueError(f"Non-finite SymPy expression cannot be converted: {node}")

        if isinstance(node, sp.Number):
            val = float(node.evalf())
            return ConstExpr(value=0.0 if abs(val) <= zero_tol else val)

        if isinstance(node, sp.Symbol):
            if node.name not in name_to_idx:
                raise ValueError(f"Unknown symbol {node.name!r} not in feature_names")
            return VarExpr(index=name_to_idx[node.name], name=node.name)

        if isinstance(node, sp.Add):
            # Order symbolic terms deterministically and place pure numbers last.
            sym_terms = [a for a in node.args if not isinstance(a, sp.Number)]
            num_terms = [a for a in node.args if isinstance(a, sp.Number)]
            sym_terms.sort(key=sp.default_sort_key)
            ordered = sym_terms + num_terms

            acc: Expr | None = None
            for arg in ordered:
                if acc is None:
                    e = _convert(arg)
                    if isinstance(e, ConstExpr) and abs(e.value) <= zero_tol:
                        continue
                    acc = e
                else:
                    is_neg, pos_arg = _split_sympy_sign(arg)
                    rhs = _convert(pos_arg)
                    if isinstance(rhs, ConstExpr) and abs(rhs.value) <= zero_tol:
                        continue
                    if is_neg:
                        acc = BinaryExpr(op="sub", left=acc, right=rhs)
                    else:
                        acc = BinaryExpr(op="add", left=acc, right=rhs)
            return acc if acc is not None else ConstExpr(value=0.0)

        if isinstance(node, sp.Mul):
            coeff, rest = node.as_coeff_Mul()
            c = float(coeff.evalf())
            if abs(c) <= zero_tol:
                return ConstExpr(value=0.0)
            if rest == sp.S.One:
                return ConstExpr(value=c)

            factors = list(rest.args) if isinstance(rest, sp.Mul) else [rest]
            num_exprs: list[Expr] = []
            den_exprs: list[Expr] = []
            for factor in factors:
                if (
                    isinstance(factor, sp.Pow)
                    and factor.exp.is_Integer
                    and int(factor.exp) < 0
                ):
                    den_e = _pow_int(_convert(factor.base), -int(factor.exp))
                    den_exprs.append(den_e)
                else:
                    fe = _convert(factor)
                    if isinstance(fe, ConstExpr) and abs(fe.value) <= zero_tol:
                        return ConstExpr(value=0.0)
                    num_exprs.append(fe)

            if not num_exprs:
                core: Expr = ConstExpr(value=abs(c))
                c = -1.0 if c < 0.0 else 1.0
            else:
                core = num_exprs[0]
                for ne in num_exprs[1:]:
                    core = BinaryExpr(op="mul", left=core, right=ne)

            for de in den_exprs:
                core = BinaryExpr(op="div", left=core, right=de)

            if abs(c - 1.0) <= 1e-9:
                return core
            if abs(c + 1.0) <= 1e-9:
                return UnaryExpr(op="neg", operand=core)
            return BinaryExpr(op="mul", left=ConstExpr(value=c), right=core)

        if isinstance(node, sp.Pow):
            base, exp = node.args
            if not exp.is_Integer:
                raise ValueError(f"Unsupported non-integer exponent in SymPy expr: {node}")
            n = int(exp)
            if n == 0:
                return ConstExpr(value=1.0)
            base_expr = _convert(base)
            if n > 0:
                return _pow_int(base_expr, n)
            return BinaryExpr(op="div", left=ConstExpr(value=1.0), right=_pow_int(base_expr, -n))

        if isinstance(node, sp.Abs):
            return UnaryExpr(op="abs", operand=_convert(node.args[0]))
        if node.func == sp.tanh:
            return UnaryExpr(op="tanh", operand=_convert(node.args[0]))
        if node.func == sp.sin:
            return UnaryExpr(op="sin", operand=_convert(node.args[0]))
        if node.func == sp.cos:
            return UnaryExpr(op="cos", operand=_convert(node.args[0]))
        if node.func == sp.Max:
            args = list(node.args)
            if len(args) == 2 and (args[0] == 0 or args[1] == 0):
                other = args[1] if args[0] == 0 else args[0]
                return UnaryExpr(op="relu", operand=_convert(other))
            acc = _convert(args[0])
            for a in args[1:]:
                acc = BinaryExpr(op="max", left=acc, right=_convert(a))
            return acc
        if node.func == sp.Min:
            args = list(node.args)
            acc = _convert(args[0])
            for a in args[1:]:
                acc = BinaryExpr(op="min", left=acc, right=_convert(a))
            return acc

        raise ValueError(f"Unsupported SymPy node type: {type(node).__name__} ({node})")

    return _convert(sym_expr)


def _normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != n:
        raise ValueError(f"weights length ({len(w)}) != sample count ({n})")
    w = np.maximum(w, 1e-12)
    mean_w = float(np.mean(w))
    return w / mean_w if mean_w > 0.0 else np.ones(n, dtype=np.float64)


def _solve_centered_ridge(
    Xc: np.ndarray,
    yc: np.ndarray,
    cols: np.ndarray,
    n_samples: int,
    alpha_small: float = 1e-9,
) -> np.ndarray:
    alpha = alpha_small if n_samples > len(cols) else 1e-3
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(Xc[:, cols], yc)
    return np.asarray(reg.coef_, dtype=np.float64)


def _prune_and_refit(
    Phi: np.ndarray,
    Phi_c: np.ndarray,
    phi_mean: np.ndarray,
    y: np.ndarray,
    yc: np.ndarray,
    y_mean: float,
    w_norm: np.ndarray,
    support: np.ndarray,
    coef_full: np.ndarray,
    threshold: float = 1e-4,
) -> tuple[np.ndarray, float, float, float]:
    """Iteratively drop negligible terms when safe and refit unpenalized support."""
    n = len(y)
    base_bias = y_mean - float(phi_mean @ coef_full)
    base_pred = Phi @ coef_full + base_bias
    base_mse = float(np.average((y - base_pred) ** 2, weights=w_norm))
    base_max = float(np.max(np.abs(y - base_pred)))

    if len(support) == 0:
        return coef_full, base_bias, base_mse, base_max

    col_scales = np.maximum(1.0, np.std(Phi, axis=0))
    kept = [int(j) for j in support if abs(coef_full[j]) >= threshold]
    dropped = sorted(
        [int(j) for j in support if abs(coef_full[j]) < threshold],
        key=lambda j: abs(coef_full[j]) * col_scales[j],
        reverse=True,
    )

    best_coef = coef_full
    best_bias = base_bias
    best_mse = base_mse
    best_max = base_max

    while True:
        cand_coef = np.zeros_like(coef_full)
        if kept:
            sel = np.asarray(kept, dtype=np.int64)
            cand_coef[sel] = _solve_centered_ridge(Phi_c, yc, sel, n, alpha_small=1e-10)
        cand_bias = y_mean - float(phi_mean @ cand_coef)
        cand_pred = Phi @ cand_coef + cand_bias
        cand_mse = float(np.average((y - cand_pred) ** 2, weights=w_norm))
        cand_max = float(np.max(np.abs(y - cand_pred)))

        safe_mse = cand_mse <= max(base_mse * 1.25 + 1e-10, 1e-8)
        safe_max = cand_max <= max(base_max * 1.25 + 1e-8, 1e-6)
        if safe_mse and safe_max:
            best_coef, best_bias, best_mse, best_max = cand_coef, cand_bias, cand_mse, cand_max
            break
        if not dropped:
            break
        # Restore the largest-impact dropped term and re-evaluate.
        kept.append(dropped.pop(0))

    if abs(best_bias) < threshold:
        no_bias_pred = Phi @ best_coef
        nb_mse = float(np.average((y - no_bias_pred) ** 2, weights=w_norm))
        nb_max = float(np.max(np.abs(y - no_bias_pred)))
        if nb_mse <= max(best_mse * 1.25 + 1e-10, 1e-8) and nb_max <= max(
            best_max * 1.25 + 1e-8, 1e-6
        ):
            best_bias = 0.0
            best_mse = nb_mse
            best_max = nb_max

    return best_coef, best_bias, best_mse, best_max


def _fit_linear_target(
    X_leaf: np.ndarray,
    y_leaf: np.ndarray,
    w_norm: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    """Fit sparse weighted linear regression w . x + b on a leaf."""
    n, d = X_leaf.shape
    if n == 0:
        return np.zeros(d, dtype=np.float64), 0.0, 0.0, 1

    y_mean = float(np.average(y_leaf, weights=w_norm))
    if n == 1 or np.allclose(y_leaf, y_mean, atol=1e-12):
        return np.zeros(d, dtype=np.float64), y_mean, 0.0, 1

    x_mean = np.average(X_leaf, axis=0, weights=w_norm)
    sqrt_w = np.sqrt(w_norm)
    Xc = (X_leaf - x_mean) * sqrt_w[:, None]
    yc = (y_leaf - y_mean) * sqrt_w

    valid_cols = np.where(np.max(np.abs(Xc), axis=0) > 1e-12)[0]
    if len(valid_cols) == 0:
        mse = float(np.average((y_leaf - y_mean) ** 2, weights=w_norm))
        return np.zeros(d, dtype=np.float64), y_mean, mse, 1

    coef_init = np.zeros(d, dtype=np.float64)
    coef_init[valid_cols] = _solve_centered_ridge(Xc, yc, valid_cols, n, alpha_small=1e-8)
    coef, bias, mse, _ = _prune_and_refit(
        X_leaf, Xc, x_mean, y_leaf, yc, y_mean, w_norm, valid_cols, coef_init
    )
    k_eff = int(np.sum(np.abs(coef) > 1e-12)) + (1 if abs(bias) > 1e-12 else 0)
    return coef, bias, mse, max(1, k_eff)


def _bic(mse: float, k: int, n: int) -> float:
    return float(n * np.log(max(mse, 1e-18)) + k * np.log(max(n, 2)))


def _fit_regression_expr(
    X_leaf: np.ndarray,
    y_leaf: np.ndarray,
    w_norm: np.ndarray,
    feature_names: Sequence[str],
    max_degree: int,
) -> Expr:
    lin_coef, lin_bias, lin_mse, lin_k = _fit_linear_target(X_leaf, y_leaf, w_norm)
    n, d = X_leaf.shape
    if max_degree < 2 or n < 6 or lin_mse <= 1e-12:
        return linear_to_expr(lin_coef, lin_bias, feature_names, zero_tol=1e-12)

    poly_deg = min(max_degree, 3 if d <= 6 else 2)
    poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
    Phi = poly.fit_transform(X_leaf)
    powers = poly.powers_
    p = Phi.shape[1]

    y_mean = float(np.average(y_leaf, weights=w_norm))
    phi_mean = np.average(Phi, axis=0, weights=w_norm)
    sqrt_w = np.sqrt(w_norm)
    Phi_c = (Phi - phi_mean) * sqrt_w[:, None]
    yc = (y_leaf - y_mean) * sqrt_w

    col_norms = np.linalg.norm(Phi_c, axis=0)
    valid_cols = np.where(col_norms > 1e-10)[0]
    if len(valid_cols) == 0:
        return linear_to_expr(lin_coef, lin_bias, feature_names, zero_tol=1e-12)

    Phi_scaled = Phi_c[:, valid_cols] / col_norms[valid_cols]
    lin_bic = _bic(lin_mse, lin_k, n)

    best_candidate: tuple[float, np.ndarray, float, float] | None = None

    def _eval_support(support: np.ndarray) -> None:
        nonlocal best_candidate
        if len(support) == 0:
            return
        coef_full = np.zeros(p, dtype=np.float64)
        coef_full[support] = _solve_centered_ridge(Phi_c, yc, support, n, alpha_small=1e-10)
        coef_p, bias_p, mse_p, _ = _prune_and_refit(
            Phi, Phi_c, phi_mean, y_leaf, yc, y_mean, w_norm, support, coef_full
        )
        nz = np.flatnonzero(np.abs(coef_p) > 1e-12)
        has_nonlinear = any(int(np.sum(powers[m])) >= 2 for m in nz)
        if not has_nonlinear:
            return
        k_eff = len(nz) + (1 if abs(bias_p) > 1e-12 else 0)
        bic_p = _bic(mse_p, max(1, k_eff), n)
        if best_candidate is None or bic_p < best_candidate[0]:
            best_candidate = (bic_p, coef_p, bias_p, mse_p)

    # Dense Ridge candidate with pruning
    _eval_support(valid_cols)

    # Sparse OMP sweep over normalized polynomial basis
    max_k = min(len(valid_cols), max(1, n // 3), 16)
    for k in range(1, max_k + 1):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            omp.fit(Phi_scaled, yc)
        sel = valid_cols[np.flatnonzero(np.abs(omp.coef_) > 1e-12)]
        _eval_support(sel)
        if best_candidate is not None and best_candidate[3] <= 1e-12:
            break

    if best_candidate is None:
        return linear_to_expr(lin_coef, lin_bias, feature_names, zero_tol=1e-12)

    best_bic, best_coef, best_bias, best_mse = best_candidate
    meaningful_improvement = (
        best_mse < lin_mse * 0.90
        and (lin_mse - best_mse) > 1e-10
        and best_bic < lin_bic
    )
    if not meaningful_improvement:
        return linear_to_expr(lin_coef, lin_bias, feature_names, zero_tol=1e-12)

    syms = [sp.Symbol(fname) for fname in feature_names]
    expr_sym: sp.Expr = sp.Float(best_bias) if abs(best_bias) > 1e-12 else sp.S.Zero
    for m in np.flatnonzero(np.abs(best_coef) > 1e-12):
        term: sp.Expr = sp.Float(float(best_coef[m]))
        for j, exp in enumerate(powers[m]):
            if int(exp) > 0:
                term = term * (syms[j] ** int(exp))
        expr_sym = expr_sym + term
    expr_sym = sp.expand(expr_sym)
    return sympy_to_expr(expr_sym, feature_names, zero_tol=1e-12)


def _sanitize_ident(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return clean or "feat"


def _format_path_condition(guard: LinearGuard, taken_true: bool) -> str:
    if taken_true:
        return guard.to_readable(precision=3)
    neg_guard = guard.model_copy(update={"op": _OP_NEGATION[guard.op]})
    return neg_guard.to_readable(precision=3)


def _leaf_direction_tag(path: list[tuple[LinearGuard, bool]]) -> str:
    if not path:
        return "global"
    guard, taken_true = path[-1]
    terms = guard.active_terms()
    if not terms:
        return "region"
    _, fname, weight = max(terms, key=lambda t: abs(t[2]))
    is_upper = guard.op in ("<=", "<")
    if not taken_true:
        is_upper = not is_upper
    if weight < 0.0:
        is_upper = not is_upper
    direction = "lo" if is_upper else "hi"
    return f"{_sanitize_ident(fname)}_{direction}"


def _prepare_targets(
    Y: np.ndarray,
    task_type: TaskType,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    Y_arr = np.asarray(Y, dtype=np.float64)
    if task_type == TaskType.REGRESSION:
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, None]
        return Y_arr, None

    # Classification: support either (N, C) logits/scores or 1D class indices
    if Y_arr.ndim == 1 or (Y_arr.ndim == 2 and Y_arr.shape[1] == 1 and output_dim >= 2):
        y_cls = Y_arr.reshape(-1).astype(np.int64)
        n = len(y_cls)
        n_cols = max(2, output_dim)
        scores = np.zeros((n, n_cols), dtype=np.float64)
        valid = (y_cls >= 0) & (y_cls < n_cols)
        scores[np.arange(n)[valid], y_cls[valid]] = 1.0
        return scores, y_cls

    if Y_arr.ndim == 2 and Y_arr.shape[1] == 1:
        y_1d = Y_arr.reshape(-1)
        if np.all((y_1d == 0.0) | (y_1d == 1.0)):
            y_cls = y_1d.astype(np.int64)
        else:
            y_cls = (y_1d > 0.5).astype(np.int64)
        return Y_arr, y_cls

    y_cls = np.argmax(Y_arr, axis=1).astype(np.int64)
    return Y_arr, y_cls


def synthesize_program(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    config: DecompileConfig,
    weights: np.ndarray | None = None,
    partition: RegimePartition | None = None,
    name: str | None = None,
) -> DecompiledProgram:
    """Synthesize a symbolic DecompiledProgram over a model's regime partition."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X array, got shape {X_arr.shape}")

    domain = model.metadata.domain
    task_type = config.task_type if config.task_type is not None else domain.task_type
    Y_target, y_cls = _prepare_targets(Y, task_type, domain.output_dim)
    if len(X_arr) != len(Y_target):
        raise ValueError(f"X rows ({len(X_arr)}) != Y rows ({len(Y_target)})")

    w_all = _normalize_weights(weights, len(X_arr))

    if partition is None:
        from glassbox.regimes import extract_regimes

        partition = extract_regimes(model, X_arr, Y_target, config, weights=w_all)

    root: RegimeNode = partition.root.model_copy(deep=True)
    assigned_ids = partition.assign(X_arr) if len(X_arr) > 0 else np.empty((0,), dtype=np.int64)
    feature_names = domain.feature_names
    out_dim = Y_target.shape[1]

    global_dominant = 0
    if y_cls is not None and len(y_cls) > 0:
        n_cls_slots = max(out_dim, int(np.max(y_cls)) + 1)
        global_counts = np.zeros(n_cls_slots, dtype=np.float64)
        np.add.at(global_counts, y_cls, w_all)
        global_dominant = int(np.argmax(global_counts))

    def _synthesize_node(node: RegimeNode, path: list[tuple[LinearGuard, bool]]) -> None:
        if isinstance(node, GuardNode):
            _synthesize_node(node.true_branch, path + [(node.guard, True)])
            _synthesize_node(node.false_branch, path + [(node.guard, False)])
            return

        assert isinstance(node, RegimeLeaf)
        idx = partition.leaf_indices.get(node.regime_id)
        if idx is None or len(idx) == 0:
            idx = np.flatnonzero(assigned_ids == node.regime_id)
        idx = np.asarray(idx, dtype=np.int64)

        node.sample_count = len(idx)
        if len(idx) == 0:
            node.output_exprs = [ConstExpr(value=0.0) for _ in range(out_dim)]
            if task_type == TaskType.CLASSIFICATION:
                node.class_id = global_dominant
                node.class_name = (
                    domain.class_names[global_dominant]
                    if global_dominant < len(domain.class_names)
                    else f"class_{global_dominant}"
                )
            else:
                node.class_id = None
                node.class_name = None
            node.local_error = 0.0
        else:
            X_leaf = X_arr[idx]
            w_leaf = w_all[idx]
            w_leaf_norm = w_leaf / max(float(np.mean(w_leaf)), 1e-12)

            if task_type == TaskType.REGRESSION:
                node.class_id = None
                node.class_name = None
                exprs = [
                    _fit_regression_expr(
                        X_leaf,
                        Y_target[idx, j],
                        w_leaf_norm,
                        feature_names,
                        config.max_degree,
                    )
                    for j in range(out_dim)
                ]
                node.output_exprs = exprs
                pred_leaf = np.array(
                    [[eval_expr(expr, row) for expr in exprs] for row in X_leaf],
                    dtype=np.float64,
                )
                node.local_error = float(
                    np.sqrt(np.average((Y_target[idx] - pred_leaf) ** 2, weights=w_leaf_norm, axis=0).mean())
                )
            else:
                assert y_cls is not None
                exprs = []
                for k in range(out_dim):
                    coef_k, bias_k, _, _ = _fit_linear_target(
                        X_leaf, Y_target[idx, k], w_leaf_norm
                    )
                    exprs.append(linear_to_expr(coef_k, bias_k, feature_names, zero_tol=1e-12))
                node.output_exprs = exprs

                y_leaf_cls = y_cls[idx]
                n_cls_slots = max(out_dim, int(np.max(y_leaf_cls)) + 1)
                cls_weights = np.zeros(n_cls_slots, dtype=np.float64)
                np.add.at(cls_weights, y_leaf_cls, w_leaf_norm)
                dominant = int(np.argmax(cls_weights))
                purity = float(cls_weights[dominant] / max(float(np.sum(cls_weights)), 1e-12))

                if len(idx) < 4 or purity >= 0.995 or out_dim == 1:
                    node.class_id = dominant
                    node.class_name = (
                        domain.class_names[dominant]
                        if dominant < len(domain.class_names)
                        else f"class_{dominant}"
                    )
                    node.local_error = float(max(0.0, 1.0 - purity))
                else:
                    node.class_id = None
                    node.class_name = None
                    pred_scores = np.array(
                        [[eval_expr(expr, row) for expr in exprs] for row in X_leaf],
                        dtype=np.float64,
                    )
                    pred_cls = np.argmax(pred_scores, axis=1)
                    acc = float(
                        np.average((pred_cls == y_leaf_cls).astype(np.float64), weights=w_leaf_norm)
                    )
                    node.local_error = float(max(0.0, 1.0 - acc))

        node.name = f"regime_{node.regime_id}_{_leaf_direction_tag(path)}"
        guard_summary = (
            " AND ".join(_format_path_condition(g, t) for g, t in path)
            if path
            else "global domain"
        )
        cls_suffix = f", class={node.class_name}" if node.class_name is not None else ""
        node.description = (
            f"{guard_summary} (samples={node.sample_count}, "
            f"local_err={node.local_error:.4g}{cls_suffix})"
        )

    _synthesize_node(root, [])

    prog_name = name or f"{model.metadata.name}_decompiled"
    program = DecompiledProgram(
        name=prog_name,
        task_type=task_type,
        domain=domain,
        root=root,
    )
    program.is_linear_ir = program.is_piecewise_linear()
    return program


__all__ = [
    "linear_to_expr",
    "sympy_to_expr",
    "synthesize_program",
]
