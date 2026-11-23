"""Counterexample-Guided Inductive Synthesis (CEGIS) decompilation loop and Pareto search."""

from __future__ import annotations

import itertools
import math
import time
import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import qmc

from glassbox.certificate import build_certificate, build_numerical_assurance
from glassbox.ir import DecompiledProgram, DecompileResult
from glassbox.types import (
    BenchmarkEntry,
    CEGISIteration,
    Counterexample,
    DecisionExplanation,
    DecompileConfig,
    DomainSpec,
    ParetoCandidate,
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
    proven = (
        2.0
        if status == VerificationStatus.PROVEN_REAL_GLOBAL
        else (1.0 if status.is_proven else 0.0)
    )
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
    """Decompile an ONNX model into a verified symbolic program via CEGIS + exact Z3 SMT."""
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
        smt_res = verify_smt(model, program, eff_cfg, iteration=t)

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
            status = smt_res.status
        elif iter_cexs:
            status = VerificationStatus.COUNTEREXAMPLE_FOUND
        elif smt_res.checked and smt_res.timed_out:
            status = VerificationStatus.TIMEOUT
        elif smt_res.checked:
            status = VerificationStatus.UNKNOWN
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

        if status.is_proven:
            break
        if not iter_cexs:
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
        real_arithmetic_proof=best_smt_res.proof,
        eval_samples=best_stats.num_samples,
    )
    return best_program, certificate, all_counterexamples


def search_pareto_frontier(
    model: LoadedModel,
    base_config: DecompileConfig | None = None,
    budgets: Sequence[int] = (2, 4, 8, 16),
    run_c_bench: bool = True,
) -> list[ParetoCandidate]:
    """Explore complexity-vs-fidelity Pareto frontier across regime budgets."""
    from glassbox.benchmark import run_benchmarks
    from glassbox.emitters import emit_all

    cfg = base_config or DecompileConfig()
    domain = model.metadata.domain
    task_type = cfg.task_type or domain.task_type
    candidates: list[ParetoCandidate] = []

    for b in budgets:
        budget = max(1, int(b))
        depth = max(1, math.ceil(math.log2(max(2, budget))) + 1)
        b_cfg = cfg.model_copy(
            update={
                "max_regimes": budget,
                "max_depth": depth,
                "max_cegis_iters": min(cfg.max_cegis_iters, 3),
                "smt_timeout_s": min(cfg.smt_timeout_s, 3.0),
            }
        )
        prog, cert, _ = decompile_with_cegis(model, b_cfg)
        code_map = emit_all(prog)

        c99_p50_us: float | None = None
        c99_native_ns: float | None = None
        if run_c_bench:
            bench_entries = run_benchmarks(model, prog, code_map, n_warmup=20, n_iters=200)
            for entry in bench_entries:
                if not entry.available:
                    continue
                if "ctypes" in entry.runtime.lower() or "dll" in entry.runtime.lower():
                    c99_p50_us = float(entry.p50_us)
                if entry.native_loop_p50_ns is not None:
                    c99_native_ns = float(entry.native_loop_p50_ns)

        candidates.append(
            ParetoCandidate(
                candidate_id=f"regimes_{budget}",
                max_regimes_budget=budget,
                actual_regimes=len(prog.leaves()),
                ast_nodes=prog.ast_node_count(),
                status=cert.status,
                agreement_rate=cert.agreement_rate,
                max_abs_error=cert.max_abs_error,
                mean_abs_error=cert.mean_abs_error,
                c99_p50_us=c99_p50_us,
                c99_native_ns=c99_native_ns,
                c_source_bytes=len(code_map.get("c", "").encode("utf-8")),
                is_pareto_optimal=False,
                program_sha256=prog.sha256(),
                code_c=code_map.get("c", ""),
                code_rust=code_map.get("rust", ""),
                code_python=code_map.get("python", ""),
            )
        )

    def _loss(cand: ParetoCandidate) -> float:
        if task_type == TaskType.CLASSIFICATION:
            return round(1.0 - cand.agreement_rate, 6)
        return round(cand.max_abs_error, 6)

    for i, cand in enumerate(candidates):
        dominated = False
        loss_i = _loss(cand)
        for j, other in enumerate(candidates):
            if i == j:
                continue
            loss_j = _loss(other)
            if (
                other.ast_nodes <= cand.ast_nodes
                and loss_j <= loss_i
                and (other.ast_nodes < cand.ast_nodes or loss_j < loss_i)
            ):
                dominated = True
                break
        cand.is_pareto_optimal = not dominated

    return candidates


def _build_default_explanations(
    model: LoadedModel,
    program: DecompiledProgram,
    counterexamples: list[Counterexample],
    leaf_proven_map: dict[int, bool],
) -> list[DecisionExplanation]:
    """Compute 5 deterministic decision explanations across domain points and counterexamples."""
    from glassbox.loader import explain_decision

    domain = model.metadata.domain
    lo = np.asarray(domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(domain.upper_bounds, dtype=np.float64)
    mid = 0.5 * (lo + hi)
    q1 = lo + 0.25 * (hi - lo)
    q3 = lo + 0.75 * (hi - lo)

    sample_pts: list[list[float]] = [mid.tolist()]
    for cex in counterexamples[:2]:
        sample_pts.append(list(cex.x))
    for pt in (q1, q3, lo + 0.15 * (hi - lo), lo + 0.85 * (hi - lo)):
        if len(sample_pts) >= 5:
            break
        sample_pts.append(pt.tolist())

    explanations: list[DecisionExplanation] = []
    for x_vec in sample_pts[:5]:
        explanations.append(
            explain_decision(model, program, x_vec, leaf_proven_map=leaf_proven_map)
        )
    return explanations


def run_decompile_pipeline(
    model: LoadedModel,
    config: DecompileConfig | None = None,
    on_iteration: Callable[[CEGISIteration], None] | None = None,
    run_id: str | None = None,
    code: dict[str, str] | None = None,
    benchmarks: list[BenchmarkEntry] | None = None,
    include_pareto: bool = False,
    pareto_budgets: Sequence[int] = (2, 4, 8, 16),
) -> DecompileResult:
    """Convenience wrapper returning a complete DecompileResult bundle with attribution."""
    from glassbox.loader import compute_regime_neuron_profiles

    program, certificate, counterexamples = decompile_with_cegis(
        model=model,
        config=config,
        on_iteration=on_iteration,
    )
    if benchmarks:
        nda = build_numerical_assurance(
            ir_vs_onnx_max_diff=certificate.max_abs_error,
            ir_vs_onnx_agreement=certificate.agreement_rate,
            onnxruntime_samples_checked=(
                certificate.numerical_deployment_assurance.onnxruntime_samples_checked
                if certificate.numerical_deployment_assurance is not None
                else 4096
            ),
            benchmarks=benchmarks,
        )
        certificate = certificate.model_copy(
            update={"benchmarks": list(benchmarks), "numerical_deployment_assurance": nda}
        )

    eff_cfg = config or DecompileConfig()
    X_attr = _sample_initial_domain(model.metadata.domain, 1024, eff_cfg.seed)
    neuron_profiles: dict[int, dict[str, Any]] = compute_regime_neuron_profiles(
        model, program, X_attr
    )
    leaf_proven_map = (
        certificate.real_arithmetic_proof.leaf_proven_map
        if certificate.real_arithmetic_proof is not None
        else {}
    )
    explanations = _build_default_explanations(model, program, counterexamples, leaf_proven_map)
    pareto = (
        search_pareto_frontier(model, eff_cfg, budgets=pareto_budgets) if include_pareto else []
    )

    return DecompileResult(
        run_id=run_id or uuid.uuid4().hex[:12],
        metadata=model.metadata,
        program=program,
        certificate=certificate,
        counterexamples=counterexamples,
        code=dict(code) if code is not None else {},
        benchmarks=list(benchmarks) if benchmarks is not None else [],
        pareto_frontier=pareto,
        regime_neuron_profiles=neuron_profiles,
        sample_explanations=explanations,
    )


__all__ = ["decompile_with_cegis", "run_decompile_pipeline", "search_pareto_frontier"]
