"""Counterexample-Guided Inductive Synthesis (CEGIS) decompilation loop."""

from __future__ import annotations

import itertools
import math
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

from glassbox.certificate import build_certificate
from glassbox.ir import DecompiledProgram, DecompileResult
from glassbox.types import (
    BenchmarkEntry,
    CEGISIteration,
    Counterexample,
    DecompileConfig,
    DomainSpec,
    TaskType,
    VerificationCertificate,
    VerificationStatus,
)
from glassbox.verifier.adversarial import EvaluationStats, evaluate_and_falsify
from glassbox.verifier.smt import SMTResult, verify_smt

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel


def _sample_initial_domain(domain: DomainSpec, n_samples: int, seed: int) -> np.ndarray:
    """Generate deterministic Sobol + uniform + corner samples across the domain box."""
    lo = np.asarray(domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(domain.upper_bounds, dtype=np.float64)
    d = domain.input_dim
    rng = np.random.default_rng(seed)

    total = max(32, int(n_samples))
    n_sobol = int(total * 0.75)
    n_uniform = max(8, total - n_sobol)

    m = max(4, math.ceil(math.log2(max(16, n_sobol))))
    sobol = qmc.Sobol(d=d, scramble=True, seed=seed)
    X_sobol = qmc.scale(sobol.random_base2(m=m)[:n_sobol], lo, hi)
    X_unif = rng.uniform(lo, hi, size=(n_uniform, d))

    midpoint = (0.5 * (lo + hi)).reshape(1, d)
    if d <= 5:
        bits = np.asarray(list(itertools.product([0.0, 1.0], repeat=d)), dtype=np.float64)
        X_corners = lo + bits * (hi - lo)
    else:
        bits = rng.integers(0, 2, size=(32, d)).astype(np.float64)
        X_corners = np.vstack([lo, hi, lo + bits * (hi - lo)])

    return np.clip(np.vstack([X_sobol, X_unif, X_corners, midpoint]), lo, hi)


def _counterexample_neighborhood(
    cexs: list[Counterexample],
    domain: DomainSpec,
    iteration: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate exact counterexample points plus local perturbations with elevated weights."""
    lo = np.asarray(domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(domain.upper_bounds, dtype=np.float64)
    span = hi - lo
    d = domain.input_dim

    selected = cexs[:16]
    pts: list[np.ndarray] = []
    wts: list[float] = []
    exact_w = 12.0 * (1.0 + 0.25 * iteration)
    near_w = 4.0 * (1.0 + 0.2 * iteration)

    for cex in selected:
        x0 = np.clip(np.asarray(cex.x, dtype=np.float64), lo, hi)
        pts.append(x0)
        wts.append(exact_w)
        for scale, count in ((0.005, 6), (0.025, 6)):
            noise = rng.normal(0.0, scale, size=(count, d)) * span
            pert = np.clip(x0 + noise, lo, hi)
            for row in pert:
                pts.append(row)
                wts.append(near_w)

    return np.asarray(pts, dtype=np.float64), np.asarray(wts, dtype=np.float64)


def _weighted_train_loss(
    program: DecompiledProgram,
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    task_type: TaskType,
) -> float:
    """Compute weighted training residual for comparing two candidate syntheses."""
    w = weights / max(float(np.sum(weights)), 1e-12)
    if task_type == TaskType.CLASSIFICATION:
        y_cls = (Y[:, 0] > 0.5).astype(np.int64) if Y.shape[1] == 1 else np.argmax(Y, axis=1)
        p_cls = program.predict_classes(X)
        return float(np.sum(w * (y_cls != p_cls).astype(np.float64)))
    pred = program.predict(X)
    return float(np.sum(w * np.max(np.abs(Y - pred), axis=1)))


def _candidate_rank(
    status: VerificationStatus,
    stats: EvaluationStats,
    max_abs_error: float,
    num_cexs: int,
    task_type: TaskType,
) -> tuple[float, ...]:
    """Higher tuple is strictly better when choosing the monotonic best program."""
    proven = 1.0 if status == VerificationStatus.PROVEN_SMT else 0.0
    zero_cex = 1.0 if num_cexs == 0 else 0.0
    if task_type == TaskType.CLASSIFICATION:
        return (proven, zero_cex, stats.agreement_rate, float(-num_cexs), -max_abs_error)
    return (proven, zero_cex, round(stats.agreement_rate, 6), -max_abs_error, -stats.rmse)


def _is_repaired(
    program: DecompiledProgram,
    cex: Counterexample,
    config: DecompileConfig,
    task_type: TaskType,
) -> bool:
    """Check whether best_program now agrees with the neural network at cex.x."""
    if task_type == TaskType.CLASSIFICATION:
        x_arr = np.asarray([cex.x], dtype=np.float64)
        prog_cls = int(program.predict_classes(x_arr)[0])
        return cex.nn_class is not None and prog_cls == cex.nn_class
    prog_y, _ = program.evaluate_one(cex.x)
    nn_y = np.asarray(cex.nn_output, dtype=np.float64)
    err = float(np.max(np.abs(nn_y - np.asarray(prog_y, dtype=np.float64))))
    return err <= config.tolerance_eps


def decompile_with_cegis(
    model: LoadedModel,
    config: DecompileConfig | None = None,
    on_iteration: Callable[[CEGISIteration], None] | None = None,
) -> tuple[DecompiledProgram, VerificationCertificate, list[Counterexample]]:
    """Decompile an ONNX model into a verified symbolic program via CEGIS + Z3 SMT."""
    from glassbox.regimes import extract_regimes, refine_regime_around_counterexamples
    from glassbox.synthesizer import synthesize_program

    base_cfg = config or DecompileConfig()
    domain = model.metadata.domain
    task_type = base_cfg.task_type or domain.task_type
    eff_cfg = base_cfg.model_copy(update={"task_type": task_type})
    rng = np.random.default_rng(eff_cfg.seed)

    X_train = _sample_initial_domain(domain, eff_cfg.initial_samples, eff_cfg.seed)
    Y_train = np.asarray(model.predict(X_train), dtype=np.float64)
    weights = np.ones(len(X_train), dtype=np.float64)

    iterations: list[CEGISIteration] = []
    all_counterexamples: list[Counterexample] = []
    active_cexs: list[Counterexample] = []

    best_program: DecompiledProgram | None = None
    best_partition = None
    best_status = VerificationStatus.UNVERIFIED
    best_stats: EvaluationStats | None = None
    best_max_err = float("inf")
    best_smt_res = SMTResult(checked=False, proven=False, timed_out=False)
    best_rank: tuple[float, ...] | None = None

    max_iters = max(1, int(eff_cfg.max_cegis_iters))
    for t in range(max_iters):
        iter_t0 = time.perf_counter()

        if t == 0 or best_partition is None:
            partition = extract_regimes(model, X_train, Y_train, eff_cfg, weights=weights)
            program = synthesize_program(
                model, X_train, Y_train, eff_cfg, weights=weights, partition=partition
            )
        else:
            X_aug, w_aug = _counterexample_neighborhood(active_cexs, domain, t, rng)
            Y_aug = np.asarray(model.predict(X_aug), dtype=np.float64)
            X_train = np.vstack([X_train, X_aug])
            Y_train = np.vstack([Y_train, Y_aug])
            weights = np.concatenate([weights, w_aug])

            cex_xs = np.asarray([c.x for c in active_cexs], dtype=np.float64)
            prev_leaf_count = len(best_partition.leaf_indices)
            partition = refine_regime_around_counterexamples(
                best_partition,
                model,
                X_train,
                Y_train,
                cex_xs,
                eff_cfg,
                weights=weights,
            )
            program = synthesize_program(
                model, X_train, Y_train, eff_cfg, weights=weights, partition=partition
            )

            # If regime budget prevented splitting, try a weighted re-partition and keep the better fit.
            if len(partition.leaf_indices) <= prev_leaf_count:
                fresh_part = extract_regimes(model, X_train, Y_train, eff_cfg, weights=weights)
                fresh_prog = synthesize_program(
                    model, X_train, Y_train, eff_cfg, weights=weights, partition=fresh_part
                )
                if _weighted_train_loss(fresh_prog, X_train, Y_train, weights, task_type) < (
                    _weighted_train_loss(program, X_train, Y_train, weights, task_type)
                ):
                    partition = fresh_part
                    program = fresh_prog

        eval_stats, adv_cexs = evaluate_and_falsify(model, program, eff_cfg, iteration=t)
        smt_res = (
            verify_smt(model, program, eff_cfg, iteration=t)
            if eff_cfg.verify_smt
            else SMTResult(checked=False, proven=False, timed_out=False, reason="smt_disabled")
        )

        iter_cexs: list[Counterexample] = []
        if smt_res.counterexample is not None:
            iter_cexs.append(smt_res.counterexample)
        for adv_c in adv_cexs:
            if smt_res.counterexample is not None:
                diff = np.max(
                    np.abs(
                        np.asarray(adv_c.x, dtype=np.float64)
                        - np.asarray(smt_res.counterexample.x, dtype=np.float64)
                    )
                )
                if diff < 1e-6:
                    continue
            iter_cexs.append(adv_c)

        iter_max_err = eval_stats.max_abs_error
        if smt_res.counterexample is not None:
            iter_max_err = max(iter_max_err, smt_res.counterexample.max_abs_error)

        if smt_res.proven:
            status = VerificationStatus.PROVEN_SMT
        elif iter_cexs:
            status = VerificationStatus.COUNTEREXAMPLE_FOUND
        else:
            status = VerificationStatus.BOUNDED_EMPIRICAL

        worst_cex = iter_cexs[0] if iter_cexs else None
        cegis_iter = CEGISIteration(
            iteration=t,
            agreement_rate=eval_stats.agreement_rate,
            max_abs_error=iter_max_err,
            mean_abs_error=eval_stats.mean_abs_error,
            rmse=eval_stats.rmse,
            regime_count=len(program.leaves()),
            ast_nodes=program.ast_node_count(),
            status=status,
            counterexamples_found=len(iter_cexs),
            worst_counterexample=worst_cex,
            smt_time_ms=smt_res.time_ms,
            total_time_ms=(time.perf_counter() - iter_t0) * 1000.0,
        )
        iterations.append(cegis_iter)
        if on_iteration is not None:
            on_iteration(cegis_iter)

        all_counterexamples.extend(iter_cexs)
        rank = _candidate_rank(status, eval_stats, iter_max_err, len(iter_cexs), task_type)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_program = program
            best_partition = partition
            best_status = status
            best_stats = eval_stats
            best_max_err = iter_max_err
            best_smt_res = smt_res

        if status == VerificationStatus.PROVEN_SMT:
            break
        if not iter_cexs:
            if task_type == TaskType.REGRESSION and iter_max_err <= eff_cfg.tolerance_eps:
                break
            if (
                task_type == TaskType.CLASSIFICATION
                and eval_stats.agreement_rate >= eff_cfg.target_agreement
            ):
                break
            # No violations left to drive refinement (e.g., SMT timed out on clean empirical fit).
            break

        active_cexs = iter_cexs

    assert best_program is not None and best_stats is not None

    for cex in all_counterexamples:
        cex.repaired = _is_repaired(best_program, cex, eff_cfg, task_type)

    certificate = build_certificate(
        model=model,
        program=best_program,
        config=eff_cfg,
        status=best_status,
        agreement_rate=best_stats.agreement_rate,
        max_abs_error=best_max_err,
        mean_abs_error=best_stats.mean_abs_error,
        smt_checked=best_smt_res.checked,
        smt_proven=best_smt_res.proven,
        total_counterexamples_found=len(all_counterexamples),
        cegis_iterations=iterations,
    )
    return best_program, certificate, all_counterexamples


def run_decompile_pipeline(
    model: LoadedModel,
    config: DecompileConfig | None = None,
    on_iteration: Callable[[CEGISIteration], None] | None = None,
    run_id: str | None = None,
    code: dict[str, str] | None = None,
    benchmarks: list[BenchmarkEntry] | None = None,
) -> DecompileResult:
    """Convenience wrapper returning a complete DecompileResult bundle."""
    program, certificate, counterexamples = decompile_with_cegis(
        model=model,
        config=config,
        on_iteration=on_iteration,
    )
    if benchmarks:
        certificate = certificate.model_copy(update={"benchmarks": list(benchmarks)})
    return DecompileResult(
        run_id=run_id or uuid.uuid4().hex[:12],
        metadata=model.metadata,
        program=program,
        certificate=certificate,
        counterexamples=counterexamples,
        code=dict(code) if code is not None else {},
        benchmarks=list(benchmarks) if benchmarks is not None else [],
    )


__all__ = ["decompile_with_cegis", "run_decompile_pipeline"]
