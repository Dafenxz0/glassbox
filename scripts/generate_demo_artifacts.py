"""Generate real pre-baked GlassBox demo artifacts and benchmark summaries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    runs_dir.mkdir(parents=True, exist_ok=True)

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

        result = run_decompile_pipeline(loaded, cfg, run_benchmarks_flag=True, run_id=mid)
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
    print(f"Generated demo artifacts for {len(model_ids)} models in {demo_dir}")


if __name__ == "__main__":
    main()
