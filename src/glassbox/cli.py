"""Command-line interface for the GlassBox neural-network-to-program decompiler."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from glassbox import __version__
from glassbox.ir import DecompiledProgram, DecompileResult
from glassbox.types import (
    CEGISIteration,
    Counterexample,
    DecompileConfig,
    DomainSpec,
    NumericalDeploymentAssurance,
    VerificationStatus,
)

app = typer.Typer(
    name="glassbox",
    help="Verifiable neural-network-to-program decompiler via CEGIS and SMT verification.",
    no_args_is_help=True,
)
zoo_app = typer.Typer(help="Manage and train built-in ONNX zoo models.", no_args_is_help=True)
app.add_typer(zoo_app, name="zoo")

console = Console()


def _default_models_dir() -> Path:
    cwd_models = Path.cwd() / "models"
    if cwd_models.is_dir():
        return cwd_models
    return Path(__file__).resolve().parents[2] / "models"


def _sample_explanation_points(
    domain: DomainSpec,
    counterexamples: list[Counterexample],
    count: int = 5,
) -> list[list[float]]:
    lo = np.asarray(domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(domain.upper_bounds, dtype=np.float64)
    mid = (0.5 * (lo + hi)).tolist()
    points: list[list[float]] = [mid]

    for cx in counterexamples:
        if len(cx.x) == domain.input_dim and len(points) < min(3, count):
            points.append([float(v) for v in cx.x])

    alphas = [0.25, 0.75, 0.35, 0.65, 0.15, 0.85]
    idx = 0
    while len(points) < count and idx < len(alphas):
        a = alphas[idx]
        idx += 1
        vec = [
            float(lo[d] + (a if d % 2 == 0 else (1.0 - a)) * (hi[d] - lo[d]))
            for d in range(domain.input_dim)
        ]
        points.append(vec)
    return points[:count]


def _enrich_numerical_assurance(result: DecompileResult) -> None:
    cert = result.certificate
    nda = cert.numerical_deployment_assurance or NumericalDeploymentAssurance(
        onnxruntime_samples_checked=200,
        ir_vs_onnx_max_diff=cert.max_abs_error,
        ir_vs_onnx_agreement=cert.agreement_rate,
    )
    for b in result.benchmarks:
        if not b.available:
            continue
        rt_lower = b.runtime.lower()
        if "c99" in rt_lower:
            nda.c99_compiled = True
            if b.compiler:
                nda.c99_compiler = b.compiler
            if nda.c99_vs_ir_max_diff is None or b.max_diff_vs_ref > nda.c99_vs_ir_max_diff:
                nda.c99_vs_ir_max_diff = float(b.max_diff_vs_ref)
        elif "python" in rt_lower:
            nda.python_vs_ir_max_diff = float(b.max_diff_vs_ref)
    cert.numerical_deployment_assurance = nda


def run_decompile_pipeline(
    model: Any,
    config: DecompileConfig,
    *,
    run_benchmarks_flag: bool = True,
    run_pareto_flag: bool = False,
    pareto_budgets: tuple[int, ...] = (2, 4, 8, 16),
    on_iteration: Any | None = None,
    run_id: str | None = None,
) -> DecompileResult:
    """Execute CEGIS decompilation, code emission, attribution, and optional benchmarking."""
    from glassbox.benchmark import run_benchmarks
    from glassbox.cegis import decompile_with_cegis
    from glassbox.emitters import emit_all

    rid = run_id or f"{model.metadata.name}_{uuid.uuid4().hex[:8]}"
    cegis_out = decompile_with_cegis(model, config, on_iteration=on_iteration)
    if isinstance(cegis_out, DecompileResult):
        result = cegis_out
        if run_id:
            result.run_id = run_id
    else:
        program, certificate, counterexamples = cegis_out
        result = DecompileResult(
            run_id=rid,
            metadata=model.metadata,
            program=program,
            certificate=certificate,
            counterexamples=counterexamples,
        )

    if not result.code:
        result.code = emit_all(result.program)
    if run_benchmarks_flag and not result.benchmarks:
        bench_entries = run_benchmarks(model, result.program, result.code)
        result.benchmarks = bench_entries
        result.certificate.benchmarks = bench_entries
    _enrich_numerical_assurance(result)

    domain = model.metadata.domain
    try:
        from glassbox.loader import compute_regime_neuron_profiles, explain_decision

        if not result.regime_neuron_profiles:
            rng = np.random.default_rng(config.seed)
            lo = np.asarray(domain.lower_bounds, dtype=np.float64)
            hi = np.asarray(domain.upper_bounds, dtype=np.float64)
            X_attr = rng.uniform(lo, hi, size=(1024, domain.input_dim))
            result.regime_neuron_profiles = compute_regime_neuron_profiles(
                model, result.program, X_attr
            )
        if not result.sample_explanations:
            leaf_map = (
                result.certificate.real_arithmetic_proof.leaf_proven_map
                if result.certificate.real_arithmetic_proof is not None
                else None
            )
            pts = _sample_explanation_points(domain, result.counterexamples, count=5)
            result.sample_explanations = [
                explain_decision(model, result.program, pt, leaf_proven_map=leaf_map)
                for pt in pts
            ]
    except ImportError:
        pass

    if run_pareto_flag and not result.pareto_frontier:
        try:
            from glassbox.cegis import search_pareto_frontier

            result.pareto_frontier = search_pareto_frontier(
                model, config, budgets=pareto_budgets, run_c_bench=run_benchmarks_flag
            )
        except ImportError:
            pass

    return result


def _status_style(status: VerificationStatus) -> str:
    if status.is_proven:
        return "bold green"
    if status == VerificationStatus.BOUNDED_EMPIRICAL:
        return "bold yellow"
    if status in {VerificationStatus.TIMEOUT, VerificationStatus.UNKNOWN}:
        return "bold magenta"
    if status == VerificationStatus.COUNTEREXAMPLE_FOUND:
        return "bold red"
    return "dim"


def _print_benchmarks_table(benchmarks: list[Any], title: str = "Runtime Benchmarks") -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Runtime")
    table.add_column("Batch", justify="right")
    table.add_column("p50 (us)", justify="right")
    table.add_column("Native (ns)", justify="right")
    table.add_column("p95 (us)", justify="right")
    table.add_column("p99 (us)", justify="right")
    table.add_column("Measured (inf/s)", justify="right")
    table.add_column("Size (B)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Max Diff", justify="right")
    table.add_column("Status")

    for b in benchmarks:
        status = "OK" if b.available else f"N/A ({b.note})"
        tput = b.measured_throughput_ips if b.measured_throughput_ips > 0 else b.throughput_ips
        native_ns = f"{b.native_loop_p50_ns:.1f}" if b.native_loop_p50_ns is not None else "-"
        table.add_row(
            b.runtime,
            str(b.batch_size),
            f"{b.p50_us:.3f}" if b.available else "-",
            native_ns if b.available else "-",
            f"{b.p95_us:.3f}" if b.available else "-",
            f"{b.p99_us:.3f}" if b.available else "-",
            f"{tput:,.0f}" if b.available else "-",
            f"{b.artifact_bytes:,}" if b.artifact_bytes else "-",
            f"{b.speedup_vs_onnx:.2f}x" if b.available else "-",
            f"{b.max_diff_vs_ref:.2e}" if b.available else "-",
            status,
        )
    console.print(table)


def _print_pareto_table(candidates: list[Any]) -> None:
    table = Table(title="Certified Program Pareto Frontier", header_style="bold cyan")
    table.add_column("ID")
    table.add_column("Budget", justify="right")
    table.add_column("Regimes", justify="right")
    table.add_column("AST Nodes", justify="right")
    table.add_column("Status")
    table.add_column("Agreement", justify="right")
    table.add_column("Max Error", justify="right")
    table.add_column("C99 (ns)", justify="right")
    table.add_column("Pareto Optimal")

    for c in candidates:
        style = _status_style(c.status)
        c_ns = f"{c.c99_native_ns:.1f}" if c.c99_native_ns is not None else "-"
        table.add_row(
            c.candidate_id,
            str(c.max_regimes_budget),
            str(c.actual_regimes),
            str(c.ast_nodes),
            f"[{style}]{c.status.value}[/{style}]",
            f"{c.agreement_rate * 100:.2f}%",
            f"{c.max_abs_error:.5f}",
            c_ns,
            "[green]YES[/green]" if c.is_pareto_optimal else "no",
        )
    console.print(table)


@app.command("inspect")
def inspect_model(
    model_path: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Path to .onnx model.")
    ],
) -> None:
    """Inspect an ONNX model's architecture, bounded domain, and piecewise-linearity."""
    from glassbox.loader import load_onnx

    loaded = load_onnx(model_path)
    meta = loaded.metadata

    summary = Table(title=f"GlassBox Model Inspection: {meta.name}", show_header=False)
    summary.add_column("Property", style="bold cyan")
    summary.add_column("Value")
    summary.add_row("Path", meta.path)
    summary.add_row("SHA-256", meta.sha256)
    summary.add_row("Size (bytes)", f"{meta.size_bytes:,}")
    summary.add_row("Task Type", meta.domain.task_type.value)
    summary.add_row("Dimensions", f"{meta.input_dim} -> {meta.output_dim}")
    summary.add_row("Total Params", f"{meta.total_params:,}")
    summary.add_row("Activations", ", ".join(meta.activations) or "none")
    summary.add_row(
        "Piecewise Linear (Tier A SMT)",
        "[green]YES[/green]" if meta.is_piecewise_linear else "[yellow]NO (Tier B)[/yellow]",
    )
    console.print(summary)

    layers_tbl = Table(title="Extracted Layers", header_style="bold cyan")
    layers_tbl.add_column("Layer")
    layers_tbl.add_column("Op")
    layers_tbl.add_column("In", justify="right")
    layers_tbl.add_column("Out", justify="right")
    layers_tbl.add_column("Activation")
    layers_tbl.add_column("Params", justify="right")
    for layer in meta.layers:
        layers_tbl.add_row(
            layer.name,
            layer.op_type,
            str(layer.in_features),
            str(layer.out_features),
            layer.activation,
            f"{layer.param_count:,}",
        )
    console.print(layers_tbl)

    dom_tbl = Table(title="Bounded Input Domain", header_style="bold cyan")
    dom_tbl.add_column("Feature")
    dom_tbl.add_column("Lower", justify="right")
    dom_tbl.add_column("Upper", justify="right")
    for fname, lo, hi in zip(
        meta.domain.feature_names, meta.domain.lower_bounds, meta.domain.upper_bounds
    ):
        dom_tbl.add_row(fname, f"{lo:.4g}", f"{hi:.4g}")
    console.print(dom_tbl)


@zoo_app.command("list")
def zoo_list(
    models_dir: Annotated[
        Path | None, typer.Option("--models-dir", help="Models directory.")
    ] = None,
) -> None:
    """List the built-in GlassBox benchmark zoo models."""
    from glassbox.zoo import list_zoo_models

    target_dir = models_dir or _default_models_dir()
    items = list_zoo_models(target_dir)

    table = Table(title=f"GlassBox Model Zoo ({target_dir})", header_style="bold cyan")
    table.add_column("ID")
    table.add_column("Task")
    table.add_column("Dims")
    table.add_column("Piecewise Linear")
    table.add_column("On Disk")
    table.add_column("Description")
    for item in items:
        meta = item.get("metadata") or {}
        domain = meta.get("domain") or item.get("domain") or {}
        in_dim = meta.get("input_dim") or len(domain.get("feature_names", []))
        out_dim = meta.get("output_dim") or len(domain.get("output_names", []))
        task = domain.get("task_type", item.get("task_type", "?"))
        pwl = meta.get("is_piecewise_linear", item.get("is_piecewise_linear", "?"))
        exists = item.get(
            "available", item.get("exists", (target_dir / f"{item['id']}.onnx").is_file())
        )
        table.add_row(
            str(item["id"]),
            str(task),
            f"{in_dim}->{out_dim}",
            str(pwl),
            "[green]YES[/green]" if exists else "[red]NO[/red]",
            str(item.get("description", ""))[:60],
        )
    console.print(table)


@zoo_app.command("train")
def zoo_train(
    force: Annotated[
        bool, typer.Option("--force", help="Retrain and overwrite existing models.")
    ] = False,
    models_dir: Annotated[
        Path | None, typer.Option("--models-dir", help="Output models directory.")
    ] = None,
) -> None:
    """Train and export all 4 built-in ONNX zoo models."""
    from glassbox.zoo import train_zoo

    target_dir = models_dir or _default_models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = train_zoo(target_dir, force=force)
    table = Table(title="Trained Zoo Models", header_style="bold cyan")
    table.add_column("Model File")
    table.add_column("Size (bytes)", justify="right")
    for p in paths:
        size = p.stat().st_size if p.is_file() else 0
        table.add_row(str(p), f"{size:,}")
    console.print(table)


@app.command("decompile")
def decompile_model(
    model_path: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Path to .onnx model.")
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output artifact directory.")
    ] = Path("out"),
    targets: Annotated[
        str,
        typer.Option("--targets", help="Comma-separated emission targets: c,rust,python,ir."),
    ] = "c,rust,python,ir",
    max_regimes: Annotated[int, typer.Option("--max-regimes", min=1, max=64)] = 8,
    max_degree: Annotated[int, typer.Option("--max-degree", min=1, max=3)] = 1,
    eps: Annotated[float, typer.Option("--eps", min=1e-6)] = 0.05,
    smt: Annotated[
        bool, typer.Option("--smt/--no-smt", help="Enable Z3 SMT verification.")
    ] = True,
    bench: Annotated[
        bool, typer.Option("--bench/--no-bench", help="Run runtime benchmarks.")
    ] = True,
    bundle: Annotated[
        bool,
        typer.Option("--bundle/--no-bundle", help="Write reproducible .gbundle directory."),
    ] = False,
    pareto: Annotated[
        bool,
        typer.Option("--pareto/--no-pareto", help="Search Pareto complexity-fidelity frontier."),
    ] = False,
) -> None:
    """Decompile an ONNX neural network via CEGIS + Z3 SMT and emit native code."""
    from glassbox.loader import load_onnx

    requested = {t.strip().lower() for t in targets.split(",") if t.strip()}
    valid = {"c", "rust", "python", "ir"}
    unknown = requested - valid
    if unknown:
        raise typer.BadParameter(f"Unknown targets: {', '.join(sorted(unknown))}")

    loaded = load_onnx(model_path)
    config = DecompileConfig(
        task_type=loaded.metadata.domain.task_type,
        max_regimes=max_regimes,
        max_degree=max_degree,
        tolerance_eps=eps,
        verify_smt=smt,
    )

    iterations_seen: list[CEGISIteration] = []
    result = run_decompile_pipeline(
        loaded,
        config,
        run_benchmarks_flag=bench,
        run_pareto_flag=pareto,
        on_iteration=iterations_seen.append,
    )

    out.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem
    written: list[Path] = []

    if "c" in requested:
        c_path = out / f"{stem}.c"
        h_path = out / f"{stem}.h"
        c_path.write_text(result.code["c"], encoding="utf-8")
        h_path.write_text(result.code["h"], encoding="utf-8")
        written.extend([c_path, h_path])
    if "rust" in requested:
        rs_path = out / f"{stem}.rs"
        rs_path.write_text(result.code["rust"], encoding="utf-8")
        written.append(rs_path)
    if "python" in requested:
        py_path = out / f"{stem}.py"
        py_path.write_text(result.code["python"], encoding="utf-8")
        written.append(py_path)
    if "ir" in requested:
        ir_path = out / f"{stem}.ir.json"
        mmd_path = out / f"{stem}.mermaid.md"
        ir_text = result.code.get("ir_json") or result.program.model_dump_json(indent=2)
        ir_path.write_text(ir_text, encoding="utf-8")
        mmd_path.write_text(result.code["mermaid"], encoding="utf-8")
        written.extend([ir_path, mmd_path])

    cert_path = out / f"{stem}.certificate.json"
    cert_path.write_text(result.certificate.model_dump_json(indent=2), encoding="utf-8")
    written.append(cert_path)

    if bundle:
        from glassbox.certificate import write_artifact_bundle

        bundle_dir = write_artifact_bundle(result, model_path, out / f"{stem}.gbundle")
        written.append(bundle_dir)

    cegis_tbl = Table(title=f"CEGIS Trace: {loaded.metadata.name}", header_style="bold cyan")
    cegis_tbl.add_column("Iter", justify="right")
    cegis_tbl.add_column("Agreement", justify="right")
    cegis_tbl.add_column("Max Error", justify="right")
    cegis_tbl.add_column("Mean Error", justify="right")
    cegis_tbl.add_column("Regimes", justify="right")
    cegis_tbl.add_column("AST Nodes", justify="right")
    cegis_tbl.add_column("CEs", justify="right")
    cegis_tbl.add_column("Status")
    cegis_tbl.add_column("SMT ms", justify="right")
    cegis_tbl.add_column("Total ms", justify="right")

    for it in result.certificate.cegis_iterations:
        style = _status_style(it.status)
        cegis_tbl.add_row(
            str(it.iteration),
            f"{it.agreement_rate * 100:.2f}%",
            f"{it.max_abs_error:.5f}",
            f"{it.mean_abs_error:.5f}",
            str(it.regime_count),
            str(it.ast_nodes),
            str(it.counterexamples_found),
            f"[{style}]{it.status.value}[/{style}]",
            f"{it.smt_time_ms:.1f}",
            f"{it.total_time_ms:.1f}",
        )
    console.print(cegis_tbl)

    cert = result.certificate
    st_style = _status_style(cert.status)
    metrics_line = (
        f"Agreement: {cert.agreement_rate * 100:.2f}% | "
        f"Max Error: {cert.max_abs_error:.6f} | "
        f"Regimes: {len(result.program.leaves())}"
    )
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Status: [{st_style}]{cert.status.value}[/{st_style}]",
                    metrics_line,
                    f"Program SHA-256: {cert.program_sha256[:16]}...",
                    f"Written ({len(written)}): {', '.join(p.name for p in written)}",
                ]
            ),
            title=f"GlassBox Verification Certificate (v{__version__})",
        )
    )

    if pareto and result.pareto_frontier:
        _print_pareto_table(result.pareto_frontier)

    if bench and result.benchmarks:
        _print_benchmarks_table(result.benchmarks)


@app.command("verify")
def verify_bundle_cmd(
    bundle_path: Annotated[
        Path, typer.Argument(exists=True, help="Path to a .gbundle directory.")
    ],
    smt: Annotated[
        bool, typer.Option("--smt/--no-smt", help="Replay exact rational Z3 SMT verification.")
    ] = True,
    timeout: Annotated[
        float, typer.Option("--timeout", min=0.5, help="Z3 SMT replay timeout in seconds.")
    ] = 6.0,
) -> None:
    """Independently verify and replay an emitted GlassBox artifact bundle (.gbundle)."""
    from glassbox.certificate import verify_artifact_bundle

    report = verify_artifact_bundle(
        bundle_path,
        recompute_smt=smt,
        smt_timeout_s=timeout,
    )

    verdict_ok = report.verdict in {"VERIFIED_REPRODUCED", "ARTIFACTS_CHECKED", "REPLAY_RESOLVED"}
    verdict_style = "bold green" if verdict_ok else "bold red"

    table = Table(
        title=f"GlassBox Artifact Bundle Verification: {bundle_path.name}",
        show_header=False,
    )
    table.add_column("Check", style="bold cyan")
    table.add_column("Result")
    table.add_row("Bundle Path", report.bundle_path)
    table.add_row("Verdict", f"[{verdict_style}]{report.verdict}[/{verdict_style}]")
    table.add_row(
        "SHA-256 Hashes Valid",
        "[green]YES[/green]" if report.hashes_valid else "[red]NO[/red]",
    )
    table.add_row("Claimed Status", report.claimed_status.value)
    table.add_row("Replayed Status", report.replayed_status.value if report.smt_replayed else "Skipped")
    table.add_row(
        "Exact Rational Z3 Replay",
        "[green]REPLAYED[/green]" if report.smt_replayed else "[yellow]SKIPPED[/yellow]",
    )
    table.add_row(
        "Numerical Deployment Checks",
        "[green]PASSED[/green]" if report.numerical_checks_passed else "[red]FAILED[/red]",
    )
    table.add_row("Replay Time (ms)", f"{report.replay_time_ms:.1f}")
    if report.details:
        table.add_row("Details", "\n".join(report.details))
    console.print(table)

    if not verdict_ok:
        raise typer.Exit(code=1)


@app.command("bench")
def bench_model(
    model_path: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Path to .onnx model.")
    ],
    ir_json_path: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Path to DecompiledProgram .ir.json.")
    ],
) -> None:
    """Benchmark an ONNX model against a decompiled program IR across runtimes."""
    from glassbox.benchmark import run_benchmarks
    from glassbox.emitters import emit_all
    from glassbox.loader import load_onnx

    loaded = load_onnx(model_path)
    program = DecompiledProgram.model_validate_json(ir_json_path.read_text(encoding="utf-8"))
    emitted = emit_all(program)
    benchmarks = run_benchmarks(loaded, program, emitted)
    _print_benchmarks_table(benchmarks, title=f"Benchmarks: {loaded.metadata.name}")


@app.command("studio")
def launch_studio(
    host: Annotated[str, typer.Option("--host", help="Bind host address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port.")] = 8765,
    models_dir: Annotated[
        Path | None, typer.Option("--models-dir", help="Models directory.")
    ] = None,
) -> None:
    """Start the GlassBox Interactive Decompiler Studio web server."""
    import uvicorn

    from glassbox.server import create_app

    fastapi_app = create_app(models_dir=models_dir)
    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
