"""Generate real pre-baked GlassBox v0.2.0 demo artifacts, Pareto frontiers, and bundles."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from glassbox.certificate import verify_artifact_bundle, write_artifact_bundle
from glassbox.cli import run_decompile_pipeline
from glassbox.loader import load_onnx
from glassbox.server import compute_slice_payload
from glassbox.types import DecompileConfig
from glassbox.zoo import train_zoo


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    demo_dir = root / "studio" / "demo-data"
    runs_dir = demo_dir / "runs"
    bundles_dir = root / "bundles"
    runs_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir.mkdir(parents=True, exist_ok=True)

    train_zoo(models_dir, force=False)

    model_ids = [
        "cartpole_policy",
        "drone_altitude_pid",
        "reactor_safety_guard",
        "nonlinear_actuator",
    ]
    index_models: list[dict[str, Any]] = []
    index_runs: dict[str, str] = {}
    summary_models: list[dict[str, Any]] = []

    for mid in model_ids:
        onnx_path = models_dir / f"{mid}.onnx"
        meta_path = models_dir / f"{mid}.meta.json"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Expected zoo model missing: {onnx_path}")

        sidecar: dict[str, Any] = {}
        if meta_path.is_file():
            sidecar = json.loads(meta_path.read_text(encoding="utf-8"))

        loaded = load_onnx(onnx_path)
        pwl = loaded.metadata.is_piecewise_linear

        raw_cfg = sidecar.get("recommended_config") or sidecar.get("default_config")
        if isinstance(raw_cfg, dict):
            cfg = DecompileConfig.model_validate(raw_cfg)
        else:
            cfg = DecompileConfig(
                task_type=loaded.metadata.domain.task_type,
                max_regimes=8,
                max_degree=1 if pwl else 2,
                tolerance_eps=0.05,
                verify_smt=pwl,
            )

        if mid == "drone_altitude_pid":
            cfg = cfg.model_copy(update={"tolerance_eps": 0.05, "smt_timeout_s": 6.0})
        elif mid == "cartpole_policy":
            cfg = cfg.model_copy(
                update={
                    "max_regimes": 16,
                    "max_depth": 4,
                    "smt_timeout_s": 8.0,
                    "classification_margin": 0.05,
                }
            )

        result = run_decompile_pipeline(
            loaded,
            cfg,
            run_benchmarks_flag=True,
            run_pareto_flag=True,
            pareto_budgets=(2, 4, 8, 16),
            run_id=mid,
        )

        # If cartpole_policy didn't reach PROVEN_REAL_MARGIN at margin=0.05, check margin=0.15
        if mid == "cartpole_policy" and not result.certificate.status.is_proven:
            alt_cfg = cfg.model_copy(update={"classification_margin": 0.15})
            alt_res = run_decompile_pipeline(
                loaded,
                alt_cfg,
                run_benchmarks_flag=True,
                run_pareto_flag=True,
                pareto_budgets=(2, 4, 8, 16),
                run_id=mid,
            )
            if alt_res.certificate.status.is_proven or (
                alt_res.certificate.agreement_rate >= result.certificate.agreement_rate
            ):
                cfg = alt_cfg
                result = alt_res

        if mid == "drone_altitude_pid":
            bundle_dir = write_artifact_bundle(
                result, onnx_path, bundles_dir / f"{mid}.gbundle"
            )
            report = verify_artifact_bundle(bundle_dir, recompute_smt=True, smt_timeout_s=6.0)
            if report.verdict != "VERIFIED_REPRODUCED":
                raise RuntimeError(
                    f"Artifact bundle self-verification failed for {mid}: {report}"
                )

        default_slice = compute_slice_payload(
            loaded,
            result,
            dim_x=0,
            dim_y=1,
            fixed_values=None,
            resolution=48,
        )

        run_file = runs_dir / f"{mid}.json"
        run_payload = {
            "result": result.model_dump(mode="json"),
            "default_slice": default_slice,
        }
        run_file.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

        index_models.append(
            {
                "id": mid,
                "name": sidecar.get("name", loaded.metadata.name),
                "description": sidecar.get("description", f"Zoo model {mid}"),
                "metadata": loaded.metadata.model_dump(mode="json"),
                "default_config": cfg.model_dump(mode="json"),
                "has_prebuilt_run": True,
            }
        )
        index_runs[mid] = f"runs/{mid}.json"

        cert = result.certificate
        rap = cert.real_arithmetic_proof
        nda = cert.numerical_deployment_assurance
        summary_models.append(
            {
                "id": mid,
                "name": sidecar.get("name", loaded.metadata.name),
                "task_type": loaded.metadata.domain.task_type.value,
                "input_dim": loaded.metadata.input_dim,
                "output_dim": loaded.metadata.output_dim,
                "total_params": loaded.metadata.total_params,
                "is_piecewise_linear": pwl,
                "status": cert.status.value,
                "smt_checked": cert.smt_checked,
                "smt_proven": cert.smt_proven,
                "agreement_rate": cert.agreement_rate,
                "max_abs_error": cert.max_abs_error,
                "mean_abs_error": cert.mean_abs_error,
                "regime_count": len(result.program.leaves()),
                "ast_nodes": result.program.ast_node_count(),
                "total_counterexamples_found": cert.total_counterexamples_found,
                "real_arithmetic_proof": rap.model_dump(mode="json") if rap else None,
                "numerical_deployment_assurance": nda.model_dump(mode="json") if nda else None,
                "pareto_candidates": len(result.pareto_frontier),
                "benchmarks": [b.model_dump(mode="json") for b in result.benchmarks],
            }
        )

    index_path = demo_dir / "index.json"
    index_path.write_text(
        json.dumps({"models": index_models, "runs": index_runs}, indent=2),
        encoding="utf-8",
    )

    summary_path = demo_dir / "benchmarks_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "models": summary_models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated v0.2.0 demo artifacts for {len(model_ids)} models in {demo_dir}")


if __name__ == "__main__":
    main()
