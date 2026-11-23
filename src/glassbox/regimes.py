"""Input-space regime partitioning, activation clustering, and counterexample repair."""

from __future__ import annotations

import heapq
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from glassbox.ir import ConstExpr, GuardNode, LinearGuard, RegimeLeaf, RegimeNode
from glassbox.loader import LoadedModel
from glassbox.types import DecompileConfig, TaskType

_MAX_ACTIVE_FEATURES = 3
_EPS = 1e-12


@dataclass
class RegimePartition:
    """Decision-tree partition of the input domain into local regimes."""

    root: GuardNode | RegimeLeaf
    leaf_indices: dict[int, np.ndarray]

    def assign(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of X to its active leaf regime_id (int64 shape (N,))."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        n = len(arr)
        if n == 0:
            return np.empty((0,), dtype=np.int64)
        out = np.empty((n,), dtype=np.int64)
        _assign_recursive(self.root, arr, np.arange(n, dtype=np.int64), out)
        return out

    def leaves(self) -> list[RegimeLeaf]:
        """Return all RegimeLeaf nodes in depth-first true-before-false order."""
        result: list[RegimeLeaf] = []

        def _walk(node: RegimeNode) -> None:
            if isinstance(node, RegimeLeaf):
                result.append(node)
            else:
                _walk(node.true_branch)
                _walk(node.false_branch)

        _walk(self.root)
        return result


def _eval_guard_mask(guard: LinearGuard, X: np.ndarray) -> np.ndarray:
    w = np.asarray(guard.weights, dtype=np.float64)
    margins = X @ w + float(guard.bias)
    if guard.op == "<=":
        return margins <= 0.0
    if guard.op == "<":
        return margins < 0.0
    if guard.op == ">=":
        return margins >= 0.0
    if guard.op == ">":
        return margins > 0.0
    raise ValueError(f"Unsupported guard op: {guard.op}")


def _assign_recursive(
    node: RegimeNode,
    X: np.ndarray,
    rows: np.ndarray,
    out: np.ndarray,
) -> None:
    if len(rows) == 0:
        return
    if isinstance(node, RegimeLeaf):
        out[rows] = int(node.regime_id)
        return
    mask = _eval_guard_mask(node.guard, X[rows])
    _assign_recursive(node.true_branch, X, rows[mask], out)
    _assign_recursive(node.false_branch, X, rows[~mask], out)


def _normalize_inputs(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    config: DecompileConfig,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TaskType]:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X array, got shape {X_arr.shape}")
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    if Y_arr.ndim != 2 or len(Y_arr) != len(X_arr):
        raise ValueError(
            f"Y shape {Y_arr.shape} is incompatible with X shape {X_arr.shape}"
        )
    if weights is None:
        w_arr = np.ones((len(X_arr),), dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w_arr) != len(X_arr):
            raise ValueError(f"weights length {len(w_arr)} != len(X) {len(X_arr)}")
        w_arr = np.maximum(w_arr, 0.0)
        if float(w_arr.sum()) <= 0.0:
            w_arr = np.ones((len(X_arr),), dtype=np.float64)

    task_type = (
        config.task_type
        if config.task_type is not None
        else model.metadata.domain.task_type
    )
    return X_arr, Y_arr, w_arr, task_type


def _y_to_classes(Y: np.ndarray) -> np.ndarray:
    if Y.shape[1] == 1:
        return (Y[:, 0] > 0.5).astype(np.int64)
    return np.argmax(Y, axis=1).astype(np.int64)


def _sparsify_direction(
    direction: np.ndarray,
    scales: np.ndarray,
    max_active: int = _MAX_ACTIVE_FEATURES,
) -> np.ndarray | None:
    d = np.asarray(direction, dtype=np.float64).copy()
    d[~np.isfinite(d)] = 0.0
    if len(d) > max_active:
        importance = np.abs(d) * np.maximum(scales, 1e-8)
        top_idx = np.argsort(importance)[::-1][:max_active]
        mask = np.zeros_like(d, dtype=bool)
        mask[top_idx] = True
        d[~mask] = 0.0
    d[np.abs(d) <= 1e-12] = 0.0
    norm = float(np.linalg.norm(d))
    if norm <= 1e-12:
        return None
    return d / norm


def _make_sparse_guard(
    weights: np.ndarray,
    bias: float,
    feature_names: list[str],
    scales: np.ndarray | None = None,
    max_active: int = _MAX_ACTIVE_FEATURES,
) -> LinearGuard:
    w = np.asarray(weights, dtype=np.float64).copy()
    w[~np.isfinite(w)] = 0.0
    if scales is None:
        scales = np.ones_like(w)
    if np.count_nonzero(np.abs(w) > 1e-12) > max_active:
        importance = np.abs(w) * np.maximum(scales, 1e-8)
        top_idx = np.argsort(importance)[::-1][:max_active]
        keep = np.zeros_like(w, dtype=bool)
        keep[top_idx] = True
        w[~keep] = 0.0
    w[np.abs(w) <= 1e-12] = 0.0
    norm = float(np.linalg.norm(w))
    b = float(bias)
    if norm > 1e-12:
        w = w / norm
        b = b / norm
    else:
        w = np.zeros_like(w)
        w[0] = 1.0
        b = 0.0
    return LinearGuard(
        weights=[float(v) for v in w],
        bias=float(b),
        feature_names=list(feature_names),
        op="<=",
    )


def _weighted_mean_mse(Y: np.ndarray, w: np.ndarray) -> float:
    w_sum = float(w.sum())
    if w_sum <= _EPS:
        return 0.0
    mean = (w[:, None] * Y).sum(axis=0) / w_sum
    diff = Y - mean
    return float((w[:, None] * (diff * diff)).sum() / (w_sum * Y.shape[1]))


def _weighted_ols_mse(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> float:
    n, d = X.shape
    w_sum = float(w.sum())
    if w_sum <= _EPS:
        return 0.0
    if n <= d + 2 or d > 16:
        return _weighted_mean_mse(Y, w)
    x_mean = (w[:, None] * X).sum(axis=0) / w_sum
    y_mean = (w[:, None] * Y).sum(axis=0) / w_sum
    Xc = X - x_mean
    Yc = Y - y_mean
    sw = np.sqrt(w)[:, None]
    Xw = Xc * sw
    Yw = Yc * sw
    xtx = Xw.T @ Xw + 1e-6 * np.eye(d, dtype=np.float64)
    xty = Xw.T @ Yw
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return _weighted_mean_mse(Y, w)
    resid = Yc - Xc @ beta
    return float((w[:, None] * (resid * resid)).sum() / (w_sum * Y.shape[1]))


def _weighted_gini(y_cls: np.ndarray, w: np.ndarray) -> float:
    w_sum = float(w.sum())
    if w_sum <= _EPS:
        return 0.0
    classes, inv = np.unique(y_cls, return_inverse=True)
    if len(classes) <= 1:
        return 0.0
    counts = np.bincount(inv, weights=w, minlength=len(classes))
    probs = counts / w_sum
    return float(1.0 - np.sum(probs * probs))


def _node_impurity(
    X: np.ndarray,
    Y: np.ndarray,
    y_cls: np.ndarray | None,
    w: np.ndarray,
    task_type: TaskType,
) -> float:
    if task_type == TaskType.CLASSIFICATION and y_cls is not None:
        return _weighted_gini(y_cls, w)
    return _weighted_ols_mse(X, Y, w)


def _candidate_split_positions(p_sorted: np.ndarray, min_samples: int, max_splits: int) -> list[int]:
    n = len(p_sorted)
    if n < 2 * min_samples:
        return []
    valid = np.flatnonzero(p_sorted[min_samples - 1 : n - min_samples] < p_sorted[min_samples : n - min_samples + 1]) + min_samples
    if len(valid) == 0:
        return []
    if len(valid) <= max_splits:
        return valid.tolist()
    picks = np.unique(np.linspace(0, len(valid) - 1, max_splits, dtype=int))
    return valid[picks].tolist()


def _scan_regression_direction(
    d_vec: np.ndarray,
    X_node: np.ndarray,
    Y_node: np.ndarray,
    w_node: np.ndarray,
    parent_imp: float,
    min_samples: int,
    max_splits: int,
    extra_threshold: float | None = None,
    prefer_exact_threshold: bool = False,
) -> tuple[float, np.ndarray, float] | None:
    proj = X_node @ d_vec
    order = np.argsort(proj, kind="mergesort")
    p_s = proj[order]
    positions = _candidate_split_positions(p_s, min_samples, max_splits)
    pos_extra: int | None = None
    if extra_threshold is not None:
        pe = int(np.searchsorted(p_s, extra_threshold, side="right"))
        if min_samples <= pe <= len(p_s) - min_samples and p_s[pe - 1] < p_s[pe]:
            pos_extra = pe
            if pe not in positions:
                positions.append(pe)
    if not positions:
        return None

    X_s = X_node[order]
    Y_s = Y_node[order]
    w_s = w_node[order]
    w_tot = float(w_s.sum())
    if w_tot <= _EPS:
        return None

    n, d = X_s.shape
    k_out = Y_s.shape[1]
    use_ols = n > 2 * (d + 2) and d <= 12

    if not use_ols:
        wy_prefix = np.cumsum(w_s[:, None] * Y_s, axis=0)
        wy2_prefix = np.cumsum(w_s[:, None] * (Y_s * Y_s), axis=0)
        w_prefix = np.cumsum(w_s)
        wy_tot = wy_prefix[-1]
        wy2_tot = wy2_prefix[-1]

        best_gain = -1.0
        best_t = 0.0
        for pos in positions:
            w_l = float(w_prefix[pos - 1])
            w_r = w_tot - w_l
            if w_l <= _EPS or w_r <= _EPS:
                continue
            sum_yl = wy_prefix[pos - 1]
            sum_yr = wy_tot - sum_yl
            sse_l = float(np.sum(wy2_prefix[pos - 1] - (sum_yl * sum_yl) / w_l))
            sse_r = float(np.sum((wy2_tot - wy2_prefix[pos - 1]) - (sum_yr * sum_yr) / w_r))
            child_imp = max(0.0, sse_l + sse_r) / (w_tot * k_out)
            gain = parent_imp - child_imp
            is_exact = (
                pos_extra is not None
                and pos == pos_extra
                and extra_threshold is not None
                and float(p_s[pos - 1]) <= float(extra_threshold) <= float(p_s[pos])
            )
            if is_exact and prefer_exact_threshold and gain > 0.0:
                gain *= 1.0 + 1e-4
            if gain > best_gain:
                best_gain = gain
                best_t = (
                    float(extra_threshold)
                    if is_exact and extra_threshold is not None
                    else 0.5 * (float(p_s[pos - 1]) + float(p_s[pos]))
                )
        if best_gain <= 1e-12:
            return None
        return w_tot * best_gain, d_vec, -best_t

    # Centered prefix sums for stable O(T * D^3) OLS MSE evaluation.
    x_mean = (w_s[:, None] * X_s).sum(axis=0) / w_tot
    y_mean = (w_s[:, None] * Y_s).sum(axis=0) / w_tot
    Xc = X_s - x_mean
    Yc = Y_s - y_mean
    Xa = np.hstack([Xc, np.ones((n, 1), dtype=np.float64)])
    da = d + 1

    w_prefix = np.cumsum(w_s)
    # (N, da, da) and (N, da, k_out) prefix sums
    wx_outer = np.cumsum(w_s[:, None, None] * (Xa[:, :, None] * Xa[:, None, :]), axis=0)
    wxy = np.cumsum(w_s[:, None, None] * (Xa[:, :, None] * Yc[:, None, :]), axis=0)
    wy2 = np.cumsum(w_s[:, None] * (Yc * Yc), axis=0)

    xtx_tot = wx_outer[-1]
    xty_tot = wxy[-1]
    y2_tot = float(np.sum(wy2[-1]))
    reg = 1e-6 * np.eye(da, dtype=np.float64)

    best_gain = -1.0
    best_t = 0.0
    for pos in positions:
        w_l = float(w_prefix[pos - 1])
        w_r = w_tot - w_l
        if w_l <= _EPS or w_r <= _EPS or pos <= d + 1 or (n - pos) <= d + 1:
            continue
        xtx_l = wx_outer[pos - 1]
        xty_l = wxy[pos - 1]
        y2_l = float(np.sum(wy2[pos - 1]))
        xtx_r = xtx_tot - xtx_l
        xty_r = xty_tot - xty_l
        y2_r = y2_tot - y2_l
        try:
            beta_l = np.linalg.solve(xtx_l + reg, xty_l)
            beta_r = np.linalg.solve(xtx_r + reg, xty_r)
        except np.linalg.LinAlgError:
            continue
        sse_l = max(0.0, y2_l - float(np.sum(beta_l * xty_l)))
        sse_r = max(0.0, y2_r - float(np.sum(beta_r * xty_r)))
        child_imp = (sse_l + sse_r) / (w_tot * k_out)
        gain = parent_imp - child_imp
        is_exact = (
            pos_extra is not None
            and pos == pos_extra
            and extra_threshold is not None
            and float(p_s[pos - 1]) <= float(extra_threshold) <= float(p_s[pos])
        )
        if is_exact and prefer_exact_threshold and gain > 0.0:
            gain *= 1.0 + 1e-4
        if gain > best_gain:
            best_gain = gain
            best_t = (
                float(extra_threshold)
                if is_exact and extra_threshold is not None
                else 0.5 * (float(p_s[pos - 1]) + float(p_s[pos]))
            )

    if best_gain <= 1e-12:
        return None
    return w_tot * best_gain, d_vec, -best_t


def _scan_classification_direction(
    d_vec: np.ndarray,
    X_node: np.ndarray,
    y_cls_node: np.ndarray,
    w_node: np.ndarray,
    parent_gini: float,
    min_samples: int,
    max_splits: int,
    extra_threshold: float | None = None,
    prefer_exact_threshold: bool = False,
) -> tuple[float, np.ndarray, float] | None:
    proj = X_node @ d_vec
    order = np.argsort(proj, kind="mergesort")
    p_s = proj[order]
    y_s = y_cls_node[order]
    w_s = w_node[order]
    w_tot = float(w_s.sum())
    if w_tot <= _EPS:
        return None

    classes, inv = np.unique(y_s, return_inverse=True)
    if len(classes) <= 1:
        return None

    n_cls = len(classes)
    onehot_w = np.zeros((len(y_s), n_cls), dtype=np.float64)
    onehot_w[np.arange(len(y_s)), inv] = w_s
    cls_prefix = np.cumsum(onehot_w, axis=0)
    cls_tot = cls_prefix[-1]
    w_prefix = np.cumsum(w_s)

    positions = _candidate_split_positions(p_s, min_samples, max_splits)
    pos_extra: int | None = None
    if extra_threshold is not None:
        pe = int(np.searchsorted(p_s, extra_threshold, side="right"))
        if min_samples <= pe <= len(p_s) - min_samples and p_s[pe - 1] < p_s[pe]:
            pos_extra = pe
            if pe not in positions:
                positions.append(pe)

    if not positions:
        return None

    best_gain = -1.0
    best_t = 0.0
    for pos in positions:
        w_l = float(w_prefix[pos - 1])
        w_r = w_tot - w_l
        if w_l <= _EPS or w_r <= _EPS:
            continue
        c_l = cls_prefix[pos - 1]
        c_r = cls_tot - c_l
        gini_l = 1.0 - float(np.sum((c_l / w_l) ** 2))
        gini_r = 1.0 - float(np.sum((c_r / w_r) ** 2))
        child_gini = (w_l * gini_l + w_r * gini_r) / w_tot
        gain = parent_gini - child_gini
        is_exact = (
            pos_extra is not None
            and pos == pos_extra
            and extra_threshold is not None
            and float(p_s[pos - 1]) <= float(extra_threshold) <= float(p_s[pos])
        )
        if is_exact and prefer_exact_threshold and gain > 0.0:
            gain *= 1.0 + 1e-4
        if gain > best_gain:
            best_gain = gain
            best_t = (
                float(extra_threshold)
                if is_exact and extra_threshold is not None
                else 0.5 * (float(p_s[pos - 1]) + float(p_s[pos]))
            )

    if best_gain <= 1e-12:
        return None
    return w_tot * best_gain, d_vec, -best_t


def _fit_sparse_classifier_direction(
    X_node: np.ndarray,
    Xs: np.ndarray,
    y_bin: np.ndarray,
    w_node: np.ndarray,
    scales: np.ndarray,
    seed: int,
    use_svc: bool = False,
) -> tuple[np.ndarray, float] | None:
    """Fit a binary linear classifier, prune to <=3 features, and refit on that subspace."""
    if len(np.unique(y_bin)) < 2:
        return None
    d = X_node.shape[1]

    def _make_clf() -> Any:
        if use_svc:
            return LinearSVC(C=2.0, max_iter=1000, random_state=seed)
        return LogisticRegression(C=4.0, max_iter=400, random_state=seed)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = _make_clf()
            clf.fit(Xs, y_bin, sample_weight=w_node)
            sp = _sparsify_direction(clf.coef_[0] / scales, scales)
            if sp is None:
                return None
            active = np.flatnonzero(np.abs(sp) > 1e-12)
            if len(active) == 0 or len(active) > _MAX_ACTIVE_FEATURES:
                return None
            sub_clf = _make_clf()
            sub_clf.fit(X_node[:, active], y_bin, sample_weight=w_node)
    except (ValueError, np.linalg.LinAlgError):
        return None

    w_full = np.zeros((d,), dtype=np.float64)
    w_full[active] = sub_clf.coef_[0]
    norm = float(np.linalg.norm(w_full))
    if norm <= 1e-12:
        return None
    return w_full / norm, -float(sub_clf.intercept_[0]) / norm


def _gather_candidate_directions(
    X_node: np.ndarray,
    Y_node: np.ndarray,
    y_cls_node: np.ndarray | None,
    w_node: np.ndarray,
    task_type: TaskType,
    seed: int,
) -> list[tuple[np.ndarray, float | None]]:
    """Return list of (sparse_unit_direction, optional_threshold) candidates."""
    n, d = X_node.shape
    w_sum = float(w_node.sum())
    x_mean = (w_node[:, None] * X_node).sum(axis=0) / w_sum
    Xc = X_node - x_mean
    var_x = (w_node[:, None] * (Xc * Xc)).sum(axis=0) / w_sum
    scales = np.sqrt(np.maximum(var_x, 1e-12))

    candidates: list[tuple[np.ndarray, float | None]] = []

    # 1. Axis-aligned directions.
    for j in range(d):
        e = np.zeros((d,), dtype=np.float64)
        e[j] = 1.0
        candidates.append((e, None))

    if d >= 2 and n >= 4:
        # 2. Weighted PCA top-2 directions (standardized so features compare fairly).
        Xs = Xc / scales
        sw = np.sqrt(w_node)[:, None]
        cov = (Xs * sw).T @ (Xs * sw) / w_sum
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            for idx in np.argsort(eigvals)[::-1][:2]:
                v = eigvecs[:, idx] / scales
                sp = _sparsify_direction(v, scales)
                if sp is not None:
                    candidates.append((sp, None))
        except np.linalg.LinAlgError:
            pass

        if task_type == TaskType.CLASSIFICATION and y_cls_node is not None:
            classes, counts = np.unique(y_cls_node, return_counts=True)
            top_classes = classes[np.argsort(counts)[::-1][:4]]
            for c in top_classes:
                mask = y_cls_node == c
                w_c = float(w_node[mask].sum())
                w_o = w_sum - w_c
                if w_c <= _EPS or w_o <= _EPS:
                    continue
                mu_c = (w_node[mask, None] * X_node[mask]).sum(axis=0) / w_c
                mu_o = (w_node[~mask, None] * X_node[~mask]).sum(axis=0) / w_o
                sp = _sparsify_direction(mu_c - mu_o, scales)
                if sp is not None:
                    candidates.append((sp, None))

                # Sparse logistic regression direction in standardized space.
                if n >= 12 and len(classes) >= 2:
                    lr_cand = _fit_sparse_classifier_direction(
                        X_node, Xs, mask.astype(np.int64), w_node, scales, seed
                    )
                    if lr_cand is not None:
                        candidates.append(lr_cand)
        else:
            # 3. Regression cross-covariance, weighted ridge slope, and residual-centroid directions.
            y_mean = (w_node[:, None] * Y_node).sum(axis=0) / w_sum
            Yc = Y_node - y_mean
            sw_x = Xc * np.sqrt(w_node)[:, None]
            sw_y = Yc * np.sqrt(w_node)[:, None]

            for k in range(min(Y_node.shape[1], 3)):
                cc = sw_x.T @ sw_y[:, k]
                sp_cc = _sparsify_direction(cc, scales)
                if sp_cc is not None:
                    candidates.append((sp_cc, None))

            xtx = sw_x.T @ sw_x + 1e-3 * np.eye(d, dtype=np.float64)
            try:
                beta = np.linalg.solve(xtx, sw_x.T @ sw_y)
                for k in range(min(Y_node.shape[1], 3)):
                    sp_b = _sparsify_direction(beta[:, k], scales)
                    if sp_b is not None:
                        candidates.append((sp_b, None))
                # Residual-polarity direction helps isolate ReLU kinks.
                resid = Yc - Xc @ beta
                r_norm = np.linalg.norm(resid, axis=1)
                med = float(np.median(r_norm))
                hi = r_norm > med
                lo = ~hi
                if np.any(hi) and np.any(lo):
                    w_hi = float(w_node[hi].sum())
                    w_lo = float(w_node[lo].sum())
                    if w_hi > _EPS and w_lo > _EPS:
                        mu_hi = (w_node[hi, None] * X_node[hi]).sum(axis=0) / w_hi
                        mu_lo = (w_node[lo, None] * X_node[lo]).sum(axis=0) / w_lo
                        sp_res = _sparsify_direction(mu_hi - mu_lo, scales)
                        if sp_res is not None:
                            candidates.append((sp_res, None))
                # Signed residual direction along primary residual component.
                r0 = resid[:, 0]
                pos_r = r0 >= 0.0
                neg_r = ~pos_r
                if np.any(pos_r) and np.any(neg_r):
                    w_p = float(w_node[pos_r].sum())
                    w_n = float(w_node[neg_r].sum())
                    if w_p > _EPS and w_n > _EPS:
                        mu_p = (w_node[pos_r, None] * X_node[pos_r]).sum(axis=0) / w_p
                        mu_n = (w_node[neg_r, None] * X_node[neg_r]).sum(axis=0) / w_n
                        sp_sr = _sparsify_direction(mu_p - mu_n, scales)
                        if sp_sr is not None:
                            candidates.append((sp_sr, None))
            except np.linalg.LinAlgError:
                pass

    return candidates


def _extract_exact_relu_hyperplanes(
    model: LoadedModel | None,
    X_node: np.ndarray,
    w_node: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Extract normalized exact Layer-1 ReLU hyperplanes and local effective hyperplanes."""
    if model is None:
        return []
    layers = model.extract_mlp_layers()
    if not layers or len(layers) < 2 or layers[0].activation != "Relu":
        return []

    hyperplanes: list[tuple[np.ndarray, float]] = []
    layer0 = layers[0]
    d = layer0.weight.shape[1]

    def _add_hp(w_raw: np.ndarray, b_raw: float) -> None:
        w_clean = np.asarray(w_raw, dtype=np.float64).copy()
        w_clean[np.abs(w_clean) <= 1e-12] = 0.0
        norm = float(np.linalg.norm(w_clean))
        if norm > 1e-12:
            hyperplanes.append((w_clean / norm, -float(b_raw) / norm))

    # 1. Exact Layer-1 neuron hyperplanes: w_j . x + b_j = 0
    for j in range(layer0.weight.shape[0]):
        _add_hp(layer0.weight[j], float(layer0.bias[j]))

    # 2. Local effective Layer-2 ReLU & output decision hyperplanes at node centroid.
    if len(X_node) >= 2:
        w_sum = float(w_node.sum())
        x_ref = (w_node[:, None] * X_node).sum(axis=0) / w_sum if w_sum > _EPS else X_node.mean(axis=0)
        z0 = X_node @ layer0.weight.T + layer0.bias
        m0_maj = (np.mean(z0 > 0.0, axis=0) >= 0.5).astype(np.float64)
        m0_ref = ((x_ref @ layer0.weight.T + layer0.bias) > 0.0).astype(np.float64)

        seen_masks: list[np.ndarray] = [m0_ref]
        if not np.array_equal(m0_maj, m0_ref):
            seen_masks.append(m0_maj)

        if len(layers) >= 3 and layers[1].activation == "Relu":
            layer1 = layers[1]
            for m0 in seen_masks:
                w_eff1 = layer1.weight @ (m0[:, None] * layer0.weight)
                b_eff1 = layer1.weight @ (m0 * layer0.bias) + layer1.bias
                for k in range(w_eff1.shape[0]):
                    _add_hp(w_eff1[k], float(b_eff1[k]))

        # Local end-to-end effective linear map for output decision boundaries
        w_cur = np.eye(d, dtype=np.float64)
        b_cur = np.zeros((d,), dtype=np.float64)
        h_cur = x_ref.copy()
        all_relu = True
        for hl in layers[:-1]:
            if hl.activation not in ("Relu", "Identity"):
                all_relu = False
                break
            z = h_cur @ hl.weight.T + hl.bias
            if hl.activation == "Relu":
                m = (z > 0.0).astype(np.float64)
                w_cur = (m[:, None] * hl.weight) @ w_cur
                b_cur = (m[:, None] * hl.weight) @ b_cur + (m * hl.bias)
                h_cur = np.maximum(0.0, z)
            else:
                w_cur = hl.weight @ w_cur
                b_cur = hl.weight @ b_cur + hl.bias
                h_cur = z
        if all_relu:
            out_l = layers[-1]
            w_out = out_l.weight @ w_cur
            b_out = out_l.weight @ b_cur + out_l.bias
            n_out = w_out.shape[0]
            if n_out == 1:
                _add_hp(w_out[0], float(b_out[0]))
            elif n_out <= 4:
                for c1 in range(n_out):
                    for c2 in range(c1 + 1, n_out):
                        _add_hp(w_out[c1] - w_out[c2], float(b_out[c1] - b_out[c2]))

    return hyperplanes


def _find_best_split(
    X: np.ndarray,
    Y: np.ndarray,
    y_cls: np.ndarray | None,
    w: np.ndarray,
    idx: np.ndarray,
    feature_names: list[str],
    task_type: TaskType,
    min_samples: int,
    seed: int,
    model: LoadedModel | None = None,
) -> tuple[float, LinearGuard, np.ndarray, np.ndarray] | None:
    if len(idx) < 2 * min_samples:
        return None
    X_node = X[idx]
    Y_node = Y[idx]
    w_node = w[idx]
    y_cls_node = y_cls[idx] if y_cls is not None else None

    parent_imp = _node_impurity(X_node, Y_node, y_cls_node, w_node, task_type)
    if parent_imp <= 1e-12:
        return None

    heur_candidates = _gather_candidate_directions(
        X_node, Y_node, y_cls_node, w_node, task_type, seed
    )
    relu_candidates = _extract_exact_relu_hyperplanes(model, X_node, w_node)

    all_candidates: list[tuple[np.ndarray, float | None, bool]] = [
        (d_vec, extra_t, False) for d_vec, extra_t in heur_candidates
    ] + [(d_vec, t_exact, True) for d_vec, t_exact in relu_candidates]

    best: tuple[float, np.ndarray, float, bool] | None = None
    for idx_cand, (d_vec, extra_t, is_relu) in enumerate(all_candidates):
        max_splits = 24 if idx_cand < X.shape[1] else 16
        if task_type == TaskType.CLASSIFICATION and y_cls_node is not None:
            res = _scan_classification_direction(
                d_vec,
                X_node,
                y_cls_node,
                w_node,
                parent_imp,
                min_samples,
                max_splits,
                extra_threshold=extra_t,
                prefer_exact_threshold=is_relu,
            )
        else:
            res = _scan_regression_direction(
                d_vec,
                X_node,
                Y_node,
                w_node,
                parent_imp,
                min_samples,
                max_splits,
                extra_threshold=extra_t,
                prefer_exact_threshold=is_relu,
            )
        if res is not None and (best is None or res[0] > best[0]):
            best = (res[0], res[1], res[2], is_relu)

    if best is None:
        return None

    score, w_dir, bias, is_relu = best
    max_active = X.shape[1] if is_relu else _MAX_ACTIVE_FEATURES
    guard = _make_sparse_guard(w_dir, bias, feature_names, max_active=max_active)
    mask = _eval_guard_mask(guard, X_node)
    left_idx = idx[mask]
    right_idx = idx[~mask]
    if len(left_idx) < min_samples or len(right_idx) < min_samples:
        return None
    return score, guard, left_idx, right_idx


@dataclass
class _PendingLeaf:
    node_id: int
    depth: int
    indices: np.ndarray
    split: tuple[float, LinearGuard, np.ndarray, np.ndarray] | None = None


def _build_leaf(
    regime_id: int,
    idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    y_cls: np.ndarray | None,
    w: np.ndarray,
    task_type: TaskType,
    class_names: list[str],
) -> RegimeLeaf:
    if len(idx) == 0:
        out_dim = Y.shape[1]
        return RegimeLeaf(
            regime_id=regime_id,
            name=f"regime_{regime_id}",
            description=f"Regime {regime_id} (empty)",
            output_exprs=[ConstExpr(value=0.0) for _ in range(out_dim)],
            sample_count=0,
            local_error=0.0,
        )

    w_node = w[idx]
    w_sum = float(w_node.sum())
    if w_sum <= _EPS:
        mean_y = Y[idx].mean(axis=0)
    else:
        mean_y = (w_node[:, None] * Y[idx]).sum(axis=0) / w_sum

    class_id: int | None = None
    class_name: str | None = None
    if task_type == TaskType.CLASSIFICATION and y_cls is not None:
        classes, inv = np.unique(y_cls[idx], return_inverse=True)
        counts = np.bincount(inv, weights=w_node, minlength=len(classes))
        class_id = int(classes[int(np.argmax(counts))])
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]

    local_err = _node_impurity(
        X[idx], Y[idx], y_cls[idx] if y_cls is not None else None, w_node, task_type
    )
    return RegimeLeaf(
        regime_id=regime_id,
        name=f"regime_{regime_id}",
        description=f"Regime {regime_id} (n={len(idx)})",
        output_exprs=[ConstExpr(value=float(v)) for v in mean_y],
        class_id=class_id,
        class_name=class_name,
        sample_count=len(idx),
        local_error=float(local_err),
    )


def _finalize_partition(
    root: RegimeNode,
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    task_type: TaskType,
    class_names: list[str],
) -> RegimePartition:
    """Assign sequential regime_ids 0..K-1 and compute exact routed leaf_indices."""
    y_cls = _y_to_classes(Y) if task_type == TaskType.CLASSIFICATION else None
    leaves_in_order: list[RegimeLeaf] = []

    def _number(node: RegimeNode) -> None:
        if isinstance(node, RegimeLeaf):
            node.regime_id = len(leaves_in_order)
            node.name = f"regime_{node.regime_id}"
            leaves_in_order.append(node)
        else:
            _number(node.true_branch)
            _number(node.false_branch)

    _number(root)

    tmp_part = RegimePartition(root=root, leaf_indices={})
    assigned = tmp_part.assign(X) if len(X) > 0 else np.empty((0,), dtype=np.int64)
    leaf_indices: dict[int, np.ndarray] = {}
    for leaf in leaves_in_order:
        rid = leaf.regime_id
        idx = np.flatnonzero(assigned == rid).astype(np.int64)
        leaf_indices[rid] = idx
        rebuilt = _build_leaf(rid, idx, X, Y, y_cls, w, task_type, class_names)
        leaf.sample_count = rebuilt.sample_count
        leaf.local_error = rebuilt.local_error
        if not leaf.output_exprs:
            leaf.output_exprs = rebuilt.output_exprs
        if leaf.class_id is None and rebuilt.class_id is not None:
            leaf.class_id = rebuilt.class_id
            leaf.class_name = rebuilt.class_name
        leaf.description = f"Regime {rid} (n={len(idx)})"

    return RegimePartition(root=root, leaf_indices=leaf_indices)


def _extract_oblique_tree(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    config: DecompileConfig,
    task_type: TaskType,
    y_cls_override: np.ndarray | None = None,
    min_samples_override: int | None = None,
) -> RegimePartition:
    n = len(X)
    feature_names = model.metadata.domain.feature_names
    class_names = model.metadata.domain.class_names
    y_cls = (
        y_cls_override
        if y_cls_override is not None
        else (_y_to_classes(Y) if task_type == TaskType.CLASSIFICATION else None)
    )
    split_task = TaskType.CLASSIFICATION if y_cls is not None else task_type
    min_samples = (
        min_samples_override if min_samples_override is not None else max(8, n // 50)
    )

    if n == 0:
        leaf = _build_leaf(0, np.empty((0,), dtype=np.int64), X, Y, y_cls, w, task_type, class_names)
        return RegimePartition(root=leaf, leaf_indices={0: np.empty((0,), dtype=np.int64)})

    max_regimes = max(1, int(config.max_regimes))
    max_depth = max(0, int(config.max_depth))

    nodes: dict[int, tuple[LinearGuard, int, int] | _PendingLeaf] = {}
    root_leaf = _PendingLeaf(node_id=0, depth=0, indices=np.arange(n, dtype=np.int64))
    nodes[0] = root_leaf
    next_id = 1
    leaf_count = 1

    heap: list[tuple[float, int]] = []
    if max_depth > 0 and max_regimes > 1:
        root_split = _find_best_split(
            X,
            Y,
            y_cls,
            w,
            root_leaf.indices,
            feature_names,
            split_task,
            min_samples,
            config.seed,
            model=model,
        )
        if root_split is not None:
            root_leaf.split = root_split
            heapq.heappush(heap, (-root_split[0], 0))

    while heap and leaf_count < max_regimes:
        _, nid = heapq.heappop(heap)
        pending = nodes[nid]
        if not isinstance(pending, _PendingLeaf) or pending.split is None:
            continue

        _, guard, left_idx, right_idx = pending.split
        left_id = next_id
        right_id = next_id + 1
        next_id += 2
        child_depth = pending.depth + 1

        left_node = _PendingLeaf(node_id=left_id, depth=child_depth, indices=left_idx)
        right_node = _PendingLeaf(node_id=right_id, depth=child_depth, indices=right_idx)
        nodes[nid] = (guard, left_id, right_id)
        nodes[left_id] = left_node
        nodes[right_id] = right_node
        leaf_count += 1

        if child_depth < max_depth and leaf_count < max_regimes:
            for child in (left_node, right_node):
                csplit = _find_best_split(
                    X,
                    Y,
                    y_cls,
                    w,
                    child.indices,
                    feature_names,
                    split_task,
                    min_samples,
                    config.seed,
                    model=model,
                )
                if csplit is not None:
                    child.split = csplit
                    heapq.heappush(heap, (-csplit[0], child.node_id))

    def _materialize(nid: int) -> RegimeNode:
        entry = nodes[nid]
        if isinstance(entry, _PendingLeaf):
            return _build_leaf(
                0,
                entry.indices,
                X,
                Y,
                _y_to_classes(Y) if task_type == TaskType.CLASSIFICATION else None,
                w,
                task_type,
                class_names,
            )
        guard, lid, rid = entry
        return GuardNode(
            guard=guard,
            true_branch=_materialize(lid),
            false_branch=_materialize(rid),
        )

    root = _materialize(0)
    return _finalize_partition(root, X, Y, w, task_type, class_names)


def _distill_clusters_to_tree(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    cluster_ids: np.ndarray,
    config: DecompileConfig,
    task_type: TaskType,
) -> RegimePartition:
    """Distill discrete activation-pattern cluster labels into a sparse LinearGuard tree."""
    n, d = X.shape
    feature_names = model.metadata.domain.feature_names
    class_names = model.metadata.domain.class_names
    min_samples = max(2, min(8, n // 50)) if n >= 16 else 1
    max_regimes = max(1, int(config.max_regimes))
    max_depth = max(0, int(config.max_depth))

    def _best_cluster_split(idx: np.ndarray) -> tuple[float, LinearGuard, np.ndarray, np.ndarray] | None:
        if len(idx) < 2 * min_samples:
            return None
        X_node = X[idx]
        c_node = cluster_ids[idx]
        w_node = w[idx]
        parent_gini = _weighted_gini(c_node, w_node)
        if parent_gini <= 1e-12:
            return None

        w_sum = float(w_node.sum())
        x_mean = (w_node[:, None] * X_node).sum(axis=0) / w_sum
        scales = np.sqrt(
            np.maximum((w_node[:, None] * ((X_node - x_mean) ** 2)).sum(axis=0) / w_sum, 1e-12)
        )
        Xs = (X_node - x_mean) / scales

        candidates: list[tuple[np.ndarray, float | None]] = []
        for j in range(d):
            e = np.zeros((d,), dtype=np.float64)
            e[j] = 1.0
            candidates.append((e, None))

        clusters, counts = np.unique(c_node, return_counts=True)
        for c in clusters[np.argsort(counts)[::-1][:4]]:
            y_bin = (c_node == c).astype(np.int64)
            for use_svc in (False, True):
                cand = _fit_sparse_classifier_direction(
                    X_node, Xs, y_bin, w_node, scales, config.seed, use_svc=use_svc
                )
                if cand is not None:
                    candidates.append(cand)

        best: tuple[float, np.ndarray, float] | None = None
        for idx_c, (d_vec, extra_t) in enumerate(candidates):
            res = _scan_classification_direction(
                d_vec,
                X_node,
                c_node,
                w_node,
                parent_gini,
                min_samples,
                24 if idx_c < d else 16,
                extra_threshold=extra_t,
            )
            if res is not None and (best is None or res[0] > best[0]):
                best = res

        if best is None:
            return None
        score, w_dir, bias = best
        guard = _make_sparse_guard(w_dir, bias, feature_names)
        mask = _eval_guard_mask(guard, X_node)
        left_idx = idx[mask]
        right_idx = idx[~mask]
        if len(left_idx) < min_samples or len(right_idx) < min_samples:
            return None
        return score, guard, left_idx, right_idx

    nodes: dict[int, tuple[LinearGuard, int, int] | _PendingLeaf] = {}
    root_leaf = _PendingLeaf(node_id=0, depth=0, indices=np.arange(n, dtype=np.int64))
    nodes[0] = root_leaf
    next_id = 1
    leaf_count = 1
    heap: list[tuple[float, int]] = []

    if max_depth > 0 and max_regimes > 1:
        split0 = _best_cluster_split(root_leaf.indices)
        if split0 is not None:
            root_leaf.split = split0
            heapq.heappush(heap, (-split0[0], 0))

    while heap and leaf_count < max_regimes:
        _, nid = heapq.heappop(heap)
        pending = nodes[nid]
        if not isinstance(pending, _PendingLeaf) or pending.split is None:
            continue
        _, guard, left_idx, right_idx = pending.split
        lid, rid = next_id, next_id + 1
        next_id += 2
        cdepth = pending.depth + 1
        lnode = _PendingLeaf(node_id=lid, depth=cdepth, indices=left_idx)
        rnode = _PendingLeaf(node_id=rid, depth=cdepth, indices=right_idx)
        nodes[nid] = (guard, lid, rid)
        nodes[lid] = lnode
        nodes[rid] = rnode
        leaf_count += 1
        if cdepth < max_depth and leaf_count < max_regimes:
            for child in (lnode, rnode):
                csplit = _best_cluster_split(child.indices)
                if csplit is not None:
                    child.split = csplit
                    heapq.heappush(heap, (-csplit[0], child.node_id))

    y_cls = _y_to_classes(Y) if task_type == TaskType.CLASSIFICATION else None

    def _materialize(nid: int) -> RegimeNode:
        entry = nodes[nid]
        if isinstance(entry, _PendingLeaf):
            return _build_leaf(0, entry.indices, X, Y, y_cls, w, task_type, class_names)
        guard, lid, rid = entry
        return GuardNode(
            guard=guard,
            true_branch=_materialize(lid),
            false_branch=_materialize(rid),
        )

    return _finalize_partition(_materialize(0), X, Y, w, task_type, class_names)


def _extract_activation_clusters(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    config: DecompileConfig,
    task_type: TaskType,
) -> RegimePartition:
    hidden_acts = model.hidden_activations(X)
    if not hidden_acts or len(X) < 2:
        return _extract_oblique_tree(model, X, Y, w, config, task_type)

    h_bin = np.hstack([(h > 0.0).astype(np.float64) for h in hidden_acts])
    y_std = Y.std(axis=0)
    y_scaled = (Y - Y.mean(axis=0)) / np.where(y_std > 1e-8, y_std, 1.0)
    y_scaled = y_scaled * (0.5 / max(1.0, float(np.sqrt(Y.shape[1]))))
    features = np.hstack([h_bin, y_scaled])

    n_unique = len(np.unique(np.round(features, 6), axis=0))
    n_clusters = min(max(1, int(config.max_regimes)), n_unique, len(X))
    if n_clusters <= 1:
        y_cls = _y_to_classes(Y) if task_type == TaskType.CLASSIFICATION else None
        leaf = _build_leaf(
            0,
            np.arange(len(X), dtype=np.int64),
            X,
            Y,
            y_cls,
            w,
            task_type,
            model.metadata.domain.class_names,
        )
        return RegimePartition(root=leaf, leaf_indices={0: np.arange(len(X), dtype=np.int64)})

    kmeans = KMeans(n_clusters=n_clusters, random_state=config.seed, n_init=10)
    cluster_ids = kmeans.fit_predict(features, sample_weight=w)
    return _distill_clusters_to_tree(model, X, Y, w, cluster_ids, config, task_type)


def extract_regimes(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    config: DecompileConfig,
    weights: np.ndarray | None = None,
) -> RegimePartition:
    """Partition input domain into sparse guard-separated regimes."""
    X_arr, Y_arr, w_arr, task_type = _normalize_inputs(model, X, Y, config, weights)
    if config.regime_method == "oblique_tree":
        return _extract_oblique_tree(model, X_arr, Y_arr, w_arr, config, task_type)
    if config.regime_method == "activation":
        return _extract_activation_clusters(model, X_arr, Y_arr, w_arr, config, task_type)
    raise ValueError(
        f"Unsupported regime_method {config.regime_method!r}; "
        "expected 'oblique_tree' or 'activation'"
    )


def _compute_leaf_residuals(
    X_leaf: np.ndarray,
    Y_leaf: np.ndarray,
    w_leaf: np.ndarray,
    task_type: TaskType,
) -> np.ndarray:
    n, d = X_leaf.shape
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    w_sum = float(w_leaf.sum())
    if task_type == TaskType.CLASSIFICATION:
        y_cls = _y_to_classes(Y_leaf)
        classes, inv = np.unique(y_cls, return_inverse=True)
        counts = np.bincount(inv, weights=w_leaf, minlength=len(classes))
        maj = classes[int(np.argmax(counts))]
        misclassified = (y_cls != maj).astype(np.float64)
        if np.any(misclassified > 0.0) and np.any(misclassified == 0.0):
            return misclassified

    if n > d + 2:
        x_mean = (w_leaf[:, None] * X_leaf).sum(axis=0) / w_sum
        y_mean = (w_leaf[:, None] * Y_leaf).sum(axis=0) / w_sum
        Xc = X_leaf - x_mean
        Yc = Y_leaf - y_mean
        sw = np.sqrt(w_leaf)[:, None]
        Xw = Xc * sw
        Yw = Yc * sw
        try:
            beta = np.linalg.solve(Xw.T @ Xw + 1e-5 * np.eye(d, dtype=np.float64), Xw.T @ Yw)
            return np.max(np.abs(Yc - Xc @ beta), axis=1)
        except np.linalg.LinAlgError:
            pass
    mean_y = (w_leaf[:, None] * Y_leaf).sum(axis=0) / w_sum
    return np.max(np.abs(Y_leaf - mean_y), axis=1)


def _split_failing_leaf(
    leaf: RegimeLeaf,
    idx: np.ndarray,
    ce_points: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    task_type: TaskType,
    seed: int,
    model: LoadedModel | None = None,
) -> GuardNode | RegimeLeaf:
    d = X.shape[1]
    y_cls = _y_to_classes(Y) if task_type == TaskType.CLASSIFICATION else None

    if len(idx) < 2:
        # Cannot split <2 points in X meaningfully; split around ce_points[0, 0] if empty.
        ce0 = ce_points[0]
        feat_idx = int(np.argmax(np.abs(ce0)))
        weights = np.zeros((d,), dtype=np.float64)
        weights[feat_idx] = 1.0
        guard = _make_sparse_guard(weights, -float(ce0[feat_idx]), feature_names)
        left_leaf = _build_leaf(0, idx, X, Y, y_cls, w, task_type, class_names)
        right_leaf = _build_leaf(0, np.empty((0,), dtype=np.int64), X, Y, y_cls, w, task_type, class_names)
        if not _eval_guard_mask(guard, X[idx]).all() if len(idx) > 0 else False:
            left_leaf, right_leaf = right_leaf, left_leaf
        return GuardNode(guard=guard, true_branch=left_leaf, false_branch=right_leaf)

    X_leaf = X[idx]
    Y_leaf = Y[idx]
    w_leaf = w[idx]
    scales = np.maximum(np.std(X_leaf, axis=0), 1e-6)
    residuals = _compute_leaf_residuals(X_leaf, Y_leaf, w_leaf, task_type)

    # Distance from each leaf point to nearest counterexample in this leaf.
    diffs = (X_leaf[:, None, :] - ce_points[None, :, :]) / scales[None, None, :]
    dists = np.min(np.linalg.norm(diffs, axis=2), axis=1)

    res_scale = max(float(np.std(residuals)), 1e-8)
    dist_scale = max(float(np.std(dists)), 1e-8)
    score = (residuals / res_scale) - 0.75 * (dists / dist_scale)

    # Mark high-residual / counterexample-proximate points as label 1.
    thresh = float(np.median(score))
    labels = (score > thresh).astype(np.int64)
    if np.all(labels == 0) or np.all(labels == 1):
        order = np.argsort(score)
        labels = np.zeros((len(idx),), dtype=np.int64)
        labels[order[len(idx) // 2 :]] = 1

    guard: LinearGuard | None = None

    # 1. First check exact ReLU hyperplanes that separate counterexamples from the leaf centroid.
    relu_hps = _extract_exact_relu_hyperplanes(model, X_leaf, w_leaf)
    if relu_hps:
        leaf_centroid = X_leaf.mean(axis=0, keepdims=True)
        parent_imp = _node_impurity(
            X_leaf,
            Y_leaf,
            y_cls[idx] if y_cls is not None else None,
            w_leaf,
            task_type,
        )
        best_relu_gain = -1.0
        for d_vec, t_exact in relu_hps:
            cand_guard = _make_sparse_guard(
                d_vec, -t_exact, feature_names, max_active=d
            )
            leaf_mask = _eval_guard_mask(cand_guard, X_leaf)
            n_left = int(leaf_mask.sum())
            if n_left == 0 or n_left == len(X_leaf):
                continue
            ce_mask = _eval_guard_mask(cand_guard, ce_points)
            cent_mask = bool(_eval_guard_mask(cand_guard, leaf_centroid)[0])
            separates_ce = bool(np.any(ce_mask != cent_mask))
            w_l = float(w_leaf[leaf_mask].sum())
            w_r = float(w_leaf[~leaf_mask].sum())
            w_tot = w_l + w_r
            if w_l <= _EPS or w_r <= _EPS:
                continue
            imp_l = _node_impurity(
                X_leaf[leaf_mask],
                Y_leaf[leaf_mask],
                y_cls[idx][leaf_mask] if y_cls is not None else None,
                w_leaf[leaf_mask],
                task_type,
            )
            imp_r = _node_impurity(
                X_leaf[~leaf_mask],
                Y_leaf[~leaf_mask],
                y_cls[idx][~leaf_mask] if y_cls is not None else None,
                w_leaf[~leaf_mask],
                task_type,
            )
            gain = parent_imp - (w_l * imp_l + w_r * imp_r) / w_tot
            if separates_ce:
                gain = max(gain, 0.0) + 1e-3 * (parent_imp + 1e-3)
            if gain > best_relu_gain and gain > 0.0:
                best_relu_gain = gain
                guard = cand_guard

    # 2. Try sparse LogisticRegression separating high-residual/CE points from low-residual points.
    if guard is None:
        X_aug = np.vstack([X_leaf, ce_points])
        y_aug = np.concatenate([labels, np.ones((len(ce_points),), dtype=np.int64)])
        w_aug = np.concatenate(
            [w_leaf, np.full((len(ce_points),), float(np.mean(w_leaf)), dtype=np.float64)]
        )
        mean_aug = X_aug.mean(axis=0)
        std_aug = np.maximum(X_aug.std(axis=0), 1e-6)
        Xs_aug = (X_aug - mean_aug) / std_aug

        lr_cand = _fit_sparse_classifier_direction(
            X_aug, Xs_aug, y_aug, w_aug, std_aug, seed, use_svc=False
        )
        if lr_cand is not None:
            w_dir, t_val = lr_cand
            cand_guard = _make_sparse_guard(w_dir, -t_val, feature_names, scales=std_aug)
            mask = _eval_guard_mask(cand_guard, X_leaf)
            if 0 < int(mask.sum()) < len(X_leaf):
                guard = cand_guard

    # 3. Fallback/comparison with best impurity-reducing split with min_samples=1.
    if guard is None:
        split = _find_best_split(
            X,
            Y,
            y_cls,
            w,
            idx,
            feature_names,
            task_type,
            min_samples=1,
            seed=seed,
            model=model,
        )
        if split is not None:
            guard = split[1]

    # 3. Deterministic centroid-difference midpoint fallback so both branches are non-empty.
    if guard is None:
        ce_mean = ce_points.mean(axis=0)
        leaf_mean = X_leaf.mean(axis=0)
        j = int(np.argmax(np.abs(ce_mean - leaf_mean) / scales))
        col = X_leaf[:, j]
        med = float(np.median(col))
        if np.all(col <= med) or np.all(col > med):
            med = 0.5 * (float(col.min()) + float(col.max()))
        w_vec = np.zeros((d,), dtype=np.float64)
        w_vec[j] = 1.0
        guard = _make_sparse_guard(w_vec, -med, feature_names)

    mask = _eval_guard_mask(guard, X_leaf)
    left_idx = idx[mask]
    right_idx = idx[~mask]
    if len(left_idx) == 0 or len(right_idx) == 0:
        # Guarantee strict partition of existing leaf indices.
        margins = X_leaf @ np.asarray(guard.weights, dtype=np.float64)
        order = np.argsort(margins, kind="mergesort")
        mid_pos = len(order) // 2
        if float(margins[order[mid_pos - 1]]) < float(margins[order[mid_pos]]):
            t = 0.5 * (float(margins[order[mid_pos - 1]]) + float(margins[order[mid_pos]]))
            guard = _make_sparse_guard(np.asarray(guard.weights), -t, feature_names)
            mask = _eval_guard_mask(guard, X_leaf)
            left_idx = idx[mask]
            right_idx = idx[~mask]
        else:
            j = int(np.argmax(np.ptp(X_leaf, axis=0) / scales))
            col = X_leaf[:, j]
            uvals = np.unique(col)
            if len(uvals) >= 2:
                mid_val = 0.5 * (float(uvals[len(uvals) // 2 - 1]) + float(uvals[len(uvals) // 2]))
                w_vec = np.zeros((d,), dtype=np.float64)
                w_vec[j] = 1.0
                guard = _make_sparse_guard(w_vec, -mid_val, feature_names)
                mask = _eval_guard_mask(guard, X_leaf)
                left_idx = idx[mask]
                right_idx = idx[~mask]
            else:
                return leaf

    left_leaf = _build_leaf(0, left_idx, X, Y, y_cls, w, task_type, class_names)
    right_leaf = _build_leaf(0, right_idx, X, Y, y_cls, w, task_type, class_names)
    return GuardNode(guard=guard, true_branch=left_leaf, false_branch=right_leaf)


def refine_regime_around_counterexamples(
    partition: RegimePartition,
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    counterexample_xs: np.ndarray,
    config: DecompileConfig,
    weights: np.ndarray | None = None,
) -> RegimePartition:
    """Surgically split leaf regimes containing counterexamples into 2 sub-regimes each."""
    X_arr, Y_arr, w_arr, task_type = _normalize_inputs(model, X, Y, config, weights)
    ce_arr = np.asarray(counterexample_xs, dtype=np.float64)
    if ce_arr.ndim == 1 and len(ce_arr) > 0:
        ce_arr = ce_arr.reshape(1, -1)
    if ce_arr.size == 0:
        return _finalize_partition(
            partition.root, X_arr, Y_arr, w_arr, task_type, model.metadata.domain.class_names
        )
    if ce_arr.ndim != 2 or ce_arr.shape[1] != X_arr.shape[1]:
        raise ValueError(
            f"counterexample_xs shape {ce_arr.shape} incompatible with X shape {X_arr.shape}"
        )

    ce_rids = partition.assign(ce_arr)
    failing_rids = sorted({int(r) for r in ce_rids})
    feature_names = model.metadata.domain.feature_names
    class_names = model.metadata.domain.class_names

    # Recompute current leaf routing on X_arr so refinement handles augmented X seamlessly.
    x_rids = partition.assign(X_arr) if len(X_arr) > 0 else np.empty((0,), dtype=np.int64)

    def _rewrite(node: RegimeNode) -> RegimeNode:
        if isinstance(node, RegimeLeaf):
            rid = int(node.regime_id)
            if rid not in failing_rids:
                return node
            leaf_idx = np.flatnonzero(x_rids == rid).astype(np.int64)
            ce_in_leaf = ce_arr[ce_rids == rid]
            return _split_failing_leaf(
                node,
                leaf_idx,
                ce_in_leaf,
                X_arr,
                Y_arr,
                w_arr,
                feature_names,
                class_names,
                task_type,
                config.seed,
                model=model,
            )
        return GuardNode(
            guard=node.guard,
            true_branch=_rewrite(node.true_branch),
            false_branch=_rewrite(node.false_branch),
        )

    new_root = _rewrite(partition.root)
    return _finalize_partition(new_root, X_arr, Y_arr, w_arr, task_type, class_names)


__all__ = [
    "RegimePartition",
    "extract_regimes",
    "refine_regime_around_counterexamples",
]
