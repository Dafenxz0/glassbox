"""Empirical evaluation and bounded adversarial falsification for decompiled programs."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.stats import qmc

from glassbox.ir import DecompiledProgram, GuardNode, LinearGuard, RegimeNode
from glassbox.types import Counterexample, DecompileConfig, DomainSpec, TaskType

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel


class EvaluationStats(BaseModel):
    """Empirical fidelity metrics over bounded domain samples."""

    agreement_rate: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    num_samples: int
    num_violations: int


def _collect_guards(node: RegimeNode) -> list[LinearGuard]:
    """Collect all LinearGuard hyperplanes from a program tree."""
    guards: list[LinearGuard] = []

    def _walk(curr: RegimeNode) -> None:
        if isinstance(curr, GuardNode):
            guards.append(curr.guard)
            _walk(curr.true_branch)
            _walk(curr.false_branch)

    _walk(node)
    return guards


def _domain_corners(domain: DomainSpec, rng: np.random.Generator) -> np.ndarray:
    """Return all corners for low-dim domains, or a deterministic subset + midpoint."""
    lo = np.asarray(domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(domain.upper_bounds, dtype=np.float64)
    d = domain.input_dim
    midpoint = (0.5 * (lo + hi)).reshape(1, d)
    if d <= 6:
        bits = np.asarray(list(itertools.product([0.0, 1.0], repeat=d)), dtype=np.float64)
        corners = lo + bits * (hi - lo)
        return np.vstack([corners, midpoint])
    bits = rng.integers(0, 2, size=(64, d)).astype(np.float64)
    corners = lo + bits * (hi - lo)
    extremes = np.vstack([lo, hi, midpoint])
    return np.vstack([corners, extremes])


def _guard_boundary_probes(
    guards: list[LinearGuard],
    base_points: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Generate points tightly straddling each GuardNode hyperplane |w.x + b| <= 0.03*||w||."""
    if not guards or len(base_points) == 0:
        return np.empty((0, lo.shape[0]), dtype=np.float64)

    probes: list[np.ndarray] = []
    per_guard = max(16, min(64, 384 // len(guards)))
    offsets = np.asarray([-0.02, -0.005, -1e-4, 1e-4, 0.005, 0.02], dtype=np.float64)

    for g_idx, guard in enumerate(guards):
        w = np.asarray(guard.weights, dtype=np.float64)
        norm_w = float(np.linalg.norm(w))
        if norm_w <= 1e-12:
            continue
        unit_w = w / norm_w
        start = (g_idx * per_guard) % len(base_points)
        idx = (np.arange(per_guard) + start) % len(base_points)
        pts = base_points[idx]
        signed_dist = (pts @ w + float(guard.bias)) / norm_w
        projected = pts - np.outer(signed_dist, unit_w)
        for off in offsets:
            cand = np.clip(projected + off * unit_w, lo, hi)
            margins = np.abs(cand @ w + float(guard.bias))
            valid = cand[margins <= 0.03 * norm_w + 1e-12]
            if len(valid) > 0:
                probes.append(valid)

    if not probes:
        return np.empty((0, lo.shape[0]), dtype=np.float64)
    return np.vstack(probes)


def _classification_margins(logits: np.ndarray) -> np.ndarray:
    """Compute top-1 vs top-2 margin per row (or |y - 0.5| for 1D outputs)."""
    if logits.shape[1] == 1:
        return np.abs(logits[:, 0] - 0.5)
    part = np.partition(logits, -2, axis=1)
    return part[:, -1] - part[:, -2]


def _single_class_margin(values: np.ndarray, class_idx: int) -> float:
    """Compute margin of class_idx over the strongest competing class."""
    if values.shape[0] == 1:
        return float(values[0] - 0.5) if class_idx == 1 else float(0.5 - values[0])
    other = max(float(values[k]) for k in range(values.shape[0]) if k != class_idx)
    return float(values[class_idx] - other)


def _select_diverse_seeds(
    X: np.ndarray,
    scores: np.ndarray,
    k: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> list[np.ndarray]:
    """Select up to k high-scoring starting points spaced apart in normalized space."""
    if k <= 0 or len(X) == 0:
        return []
    order = np.argsort(scores)[::-1]
    span = np.maximum(hi - lo, 1e-12)
    selected: list[np.ndarray] = []
    selected_norm: list[np.ndarray] = []
    for idx in order:
        x_norm = (X[idx] - lo) / span
        if all(float(np.max(np.abs(x_norm - prev))) > 0.03 for prev in selected_norm):
            selected.append(X[idx].copy())
            selected_norm.append(x_norm)
            if len(selected) >= k:
                break
    if len(selected) < k:
        for idx in order:
            selected.append(X[idx].copy())
            if len(selected) >= k:
                break
    return selected


def _run_adversarial_restarts(
    model: LoadedModel,
    program: DecompiledProgram,
    seeds: list[np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    task_type: TaskType,
) -> np.ndarray:
    """Run bounded derivative-free multi-start optimization maximizing discrepancy."""
    if not seeds:
        return np.empty((0, lo.shape[0]), dtype=np.float64)

    bounds = list(zip(lo.tolist(), hi.tolist()))
    best_points: list[np.ndarray] = []

    if task_type == TaskType.REGRESSION:

        def objective(x_vec: np.ndarray) -> float:
            xc = np.clip(x_vec, lo, hi)
            nn_y = model.predict(xc.reshape(1, -1))[0]
            prog_y, _ = program.evaluate_one(xc.tolist())
            return -float(np.max(np.abs(nn_y - np.asarray(prog_y, dtype=np.float64))))

    else:

        def objective(x_vec: np.ndarray) -> float:
            xc = np.clip(x_vec, lo, hi)
            x2d = xc.reshape(1, -1)
            nn_y = model.predict(x2d)[0]
            nn_cls = int(model.predict_classes(x2d)[0])
            leaf = program._route_leaf(xc.tolist())
            prog_y_list, _ = program.evaluate_one(xc.tolist())
            prog_y = np.asarray(prog_y_list, dtype=np.float64)
            if leaf.class_id is not None:
                prog_cls = int(leaf.class_id)
                prog_margin_c1 = 1.0 if prog_cls == nn_cls else -1.0
            else:
                prog_cls = (
                    int(prog_y[0] > 0.5) if prog_y.shape[0] == 1 else int(np.argmax(prog_y))
                )
                prog_margin_c1 = _single_class_margin(prog_y, nn_cls)
            nn_margin_c1 = _single_class_margin(nn_y, nn_cls)
            bonus = 10.0 if prog_cls != nn_cls else 0.0
            return -(bonus + nn_margin_c1 - prog_margin_c1)

    for x0 in seeds:
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxfev": 60, "xatol": 1e-4, "fatol": 1e-4},
        )
        best_points.append(np.clip(np.asarray(res.x, dtype=np.float64), lo, hi))

    return np.asarray(best_points, dtype=np.float64)


def _deduplicate_counterexamples(
    candidates: list[tuple[float, Counterexample]],
    lo: np.ndarray,
    hi: np.ndarray,
    max_count: int = 32,
) -> list[Counterexample]:
    """Sort counterexamples by descending severity and deduplicate nearby points."""
    candidates.sort(key=lambda item: item[0], reverse=True)
    span = np.maximum(hi - lo, 1e-12)
    kept: list[Counterexample] = []
    kept_norm: list[np.ndarray] = []

    for _, cex in candidates:
        x_norm = (np.asarray(cex.x, dtype=np.float64) - lo) / span
        if any(
            float(np.max(np.abs(x_norm - prev))) < 1e-2 and cex.regime_id == prev_cex.regime_id
            for prev, prev_cex in zip(kept_norm, kept)
        ):
            continue
        kept.append(cex)
        kept_norm.append(x_norm)
        if len(kept) >= max_count:
            break
    return kept


def evaluate_and_falsify(
    model: LoadedModel,
    program: DecompiledProgram,
    config: DecompileConfig,
    iteration: int = 0,
) -> tuple[EvaluationStats, list[Counterexample]]:
    """Evaluate empirical fidelity over >=4096 samples and search for worst counterexamples."""
    domain = model.metadata.domain
    task_type = config.task_type or domain.task_type
    lo = np.asarray(domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(domain.upper_bounds, dtype=np.float64)
    d = domain.input_dim

    seed = config.seed + 1009 * (iteration + 1)
    rng = np.random.default_rng(seed)

    sobol = qmc.Sobol(d=d, scramble=True, seed=seed)
    X_sobol = qmc.scale(sobol.random_base2(m=12), lo, hi)
    X_corners = _domain_corners(domain, rng)
    guards = _collect_guards(program.root)
    X_boundary = _guard_boundary_probes(guards, X_sobol, lo, hi)

    blocks = [X_sobol, X_corners]
    if len(X_boundary) > 0:
        blocks.append(X_boundary)
    X_eval = np.clip(np.vstack(blocks), lo, hi)

    nn_Y = np.asarray(model.predict(X_eval), dtype=np.float64)
    prog_Y, _ = program.predict_with_regimes(X_eval)
    abs_err = np.abs(nn_Y - prog_Y)
    row_max_err = np.max(abs_err, axis=1)

    max_abs_error = float(np.max(row_max_err))
    mean_abs_error = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((nn_Y - prog_Y) ** 2)))

    if task_type == TaskType.REGRESSION:
        violated_mask = row_max_err > config.tolerance_eps
        agreement_rate = float(np.mean(~violated_mask))
        seed_scores = row_max_err
        nn_cls_eval = None
        prog_cls_eval = None
        nn_margins_eval = None
    else:
        nn_cls_eval = np.asarray(model.predict_classes(X_eval), dtype=np.int64)
        prog_cls_eval = np.asarray(program.predict_classes(X_eval), dtype=np.int64)
        nn_margins_eval = _classification_margins(nn_Y)
        disagree = nn_cls_eval != prog_cls_eval
        agreement_rate = float(np.mean(~disagree))
        violated_mask = disagree & (nn_margins_eval >= config.classification_margin)
        # Prioritize disagreements with strong margin, then low-margin boundary points.
        seed_scores = np.where(disagree, 10.0 + nn_margins_eval, -nn_margins_eval + 0.1 * row_max_err)

    restarts = max(0, int(config.adversarial_restarts))
    seeds = _select_diverse_seeds(X_eval, seed_scores, restarts, lo, hi)
    X_opt = _run_adversarial_restarts(model, program, seeds, lo, hi, task_type)

    # Pool top empirical violations + all adversarially optimized candidates.
    viol_indices = np.flatnonzero(violated_mask)
    if len(viol_indices) > 64:
        top_local = np.argsort(seed_scores[viol_indices])[::-1][:64]
        viol_indices = viol_indices[top_local]

    candidate_blocks = []
    if len(viol_indices) > 0:
        candidate_blocks.append(X_eval[viol_indices])
    if len(X_opt) > 0:
        candidate_blocks.append(X_opt)

    scored_cexs: list[tuple[float, Counterexample]] = []
    if candidate_blocks:
        X_cand = np.clip(np.vstack(candidate_blocks), lo, hi)
        cand_nn_Y = np.asarray(model.predict(X_cand), dtype=np.float64)
        cand_prog_Y, cand_regimes = program.predict_with_regimes(X_cand)
        cand_max_err = np.max(np.abs(cand_nn_Y - cand_prog_Y), axis=1)
        max_abs_error = max(max_abs_error, float(np.max(cand_max_err)))

        if task_type == TaskType.REGRESSION:
            for i in range(len(X_cand)):
                err = float(cand_max_err[i])
                if err > config.tolerance_eps:
                    scored_cexs.append(
                        (
                            err,
                            Counterexample(
                                iteration=iteration,
                                source="adversarial",
                                x=X_cand[i].tolist(),
                                nn_output=cand_nn_Y[i].tolist(),
                                program_output=cand_prog_Y[i].tolist(),
                                nn_class=None,
                                program_class=None,
                                max_abs_error=err,
                                regime_id=int(cand_regimes[i]),
                                repaired=False,
                            ),
                        )
                    )
        else:
            cand_nn_cls = np.asarray(model.predict_classes(X_cand), dtype=np.int64)
            cand_prog_cls = np.asarray(program.predict_classes(X_cand), dtype=np.int64)
            cand_margins = _classification_margins(cand_nn_Y)
            for i in range(len(X_cand)):
                if (
                    int(cand_nn_cls[i]) != int(cand_prog_cls[i])
                    and float(cand_margins[i]) >= config.classification_margin
                ):
                    severity = float(cand_margins[i]) * 100.0 + float(cand_max_err[i])
                    scored_cexs.append(
                        (
                            severity,
                            Counterexample(
                                iteration=iteration,
                                source="adversarial",
                                x=X_cand[i].tolist(),
                                nn_output=cand_nn_Y[i].tolist(),
                                program_output=cand_prog_Y[i].tolist(),
                                nn_class=int(cand_nn_cls[i]),
                                program_class=int(cand_prog_cls[i]),
                                max_abs_error=float(cand_max_err[i]),
                                regime_id=int(cand_regimes[i]),
                                repaired=False,
                            ),
                        )
                    )

    counterexamples = _deduplicate_counterexamples(scored_cexs, lo, hi, max_count=32)
    stats = EvaluationStats(
        agreement_rate=agreement_rate,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        rmse=rmse,
        num_samples=len(X_eval),
        num_violations=int(np.sum(violated_mask)),
    )
    return stats, counterexamples


__all__ = ["EvaluationStats", "evaluate_and_falsify"]
