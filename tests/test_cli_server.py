"""Tests for GlassBox v0.2.0 CLI commands and FastAPI Studio server endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from torch import nn
from typer.testing import CliRunner

from glassbox.cli import app as cli_app
from glassbox.loader import LoadedModel
from glassbox.server import MAX_UPLOAD_BYTES, create_app
from glassbox.types import DomainSpec, TaskType


def _export_tiny_relu_model(tmp_path: Path, name: str = "tiny_ctrl") -> Path:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(2, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )
    domain = DomainSpec(
        feature_names=["pos", "vel"],
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        output_names=["cmd"],
        task_type=TaskType.REGRESSION,
    )
    onnx_path = tmp_path / f"{name}.onnx"
    LoadedModel.from_pytorch(net, domain, onnx_path, name=name)
    return onnx_path


def test_api_health_and_models(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _export_tiny_relu_model(models_dir, "tiny_ctrl")

    app = create_app(
        models_dir=models_dir,
        studio_dir=tmp_path / "no_studio",
        site_dir=tmp_path / "no_site",
    )
    client = TestClient(app)

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    body = res_health.json()
    assert body["status"] == "ok"
    assert body["z3_available"] is True

    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    models = res_models.json()
    assert len(models) == 1
    assert models[0]["id"] == "tiny_ctrl"
    assert models[0]["metadata"]["is_piecewise_linear"] is True


def test_api_decompile_explain_slice_and_download(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _export_tiny_relu_model(models_dir, "tiny_ctrl")

    app = create_app(
        models_dir=models_dir,
        studio_dir=tmp_path / "no_studio",
        site_dir=tmp_path / "no_site",
    )
    client = TestClient(app)

    res_dec = client.post(
        "/api/decompile",
        json={
            "model_id": "tiny_ctrl",
            "config": {
                "max_regimes": 6,
                "max_degree": 1,
                "tolerance_eps": 0.1,
                "max_cegis_iters": 3,
                "initial_samples": 400,
                "verify_smt": True,
                "smt_timeout_s": 2.0,
            },
            "run_benchmarks": False,
        },
    )
    assert res_dec.status_code == 200, res_dec.text
    dec = res_dec.json()
    run_id = dec["run_id"]
    assert "c" in dec["code"]
    assert "rust" in dec["code"]
    assert "regime_neuron_profiles" in dec
    assert "sample_explanations" in dec

    res_run = client.get(f"/api/runs/{run_id}")
    assert res_run.status_code == 200
    run_body = res_run.json()
    assert run_body["run_id"] == run_id
    assert "certificate" in run_body

    res_explain = client.post(
        "/api/explain",
        json={"run_id": run_id, "x": [0.25, -0.25]},
    )
    assert res_explain.status_code == 200, res_explain.text
    exp = res_explain.json()
    assert exp["x"] == [0.25, -0.25]
    assert "regime_id" in exp
    assert "guard_trace" in exp
    assert "summary_text" in exp

    res_bundle = client.get(f"/api/runs/{run_id}/bundle")
    assert res_bundle.status_code == 200
    assert res_bundle.json()["run_id"] == run_id

    res_slice = client.post(
        "/api/slice",
        json={
            "run_id": run_id,
            "dim_x": 0,
            "dim_y": 1,
            "fixed_values": [0.0, 0.0],
            "resolution": 16,
        },
    )
    assert res_slice.status_code == 200, res_slice.text
    sl = res_slice.json()
    assert sl["resolution"] == 16
    assert len(sl["nn_grid"]) == 16
    assert len(sl["nn_grid"][0]) == 16
    assert len(sl["diff_grid"]) == 16
    assert len(sl["regime_grid"]) == 16

    res_dl = client.get(f"/api/runs/{run_id}/download/c")
    assert res_dl.status_code == 200
    assert "attachment;" in res_dl.headers["content-disposition"]
    assert "glassbox_infer" in res_dl.text


def test_api_upload_accepts_valid_and_rejects_invalid(tmp_path: Path) -> None:
    app = create_app(
        models_dir=tmp_path / "models",
        studio_dir=tmp_path / "no_studio",
        site_dir=tmp_path / "no_site",
    )
    client = TestClient(app)

    bad_proto = client.post(
        "/api/upload",
        files={"file": ("corrupt.onnx", b"not-an-onnx-protobuf", "application/octet-stream")},
    )
    assert bad_proto.status_code == 400

    bad_ext = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad_ext.status_code == 400

    oversize = client.post(
        "/api/upload",
        files={"file": ("huge.onnx", b"\x00" * (MAX_UPLOAD_BYTES + 10), "application/octet-stream")},
    )
    assert oversize.status_code == 400

    onnx_path = _export_tiny_relu_model(tmp_path, "uploaded_ctrl")
    good = client.post(
        "/api/upload",
        files={"file": ("uploaded_ctrl.onnx", onnx_path.read_bytes(), "application/octet-stream")},
    )
    assert good.status_code == 200, good.text
    assert good.json()["model_id"] == "uploaded_ctrl"


def test_api_stream_sse(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _export_tiny_relu_model(models_dir, "tiny_ctrl")

    app = create_app(
        models_dir=models_dir,
        studio_dir=tmp_path / "no_studio",
        site_dir=tmp_path / "no_site",
    )
    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/decompile/stream",
        json={
            "model_id": "tiny_ctrl",
            "config": {
                "max_regimes": 4,
                "max_cegis_iters": 2,
                "initial_samples": 300,
                "verify_smt": True,
            },
            "run_benchmarks": False,
        },
    ) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "event: iteration" in text
    assert "event: complete" in text


def test_cli_inspect_decompile_bundle_and_verify(tmp_path: Path) -> None:
    onnx_path = _export_tiny_relu_model(tmp_path, "cli_ctrl")
    out_dir = tmp_path / "out"
    runner = CliRunner()

    res_inspect = runner.invoke(cli_app, ["inspect", str(onnx_path)])
    assert res_inspect.exit_code == 0, res_inspect.output
    assert "cli_ctrl" in res_inspect.output

    res_dec = runner.invoke(
        cli_app,
        [
            "decompile",
            str(onnx_path),
            "--out",
            str(out_dir),
            "--targets",
            "c,rust,python,ir",
            "--max-regimes",
            "6",
            "--eps",
            "0.25",
            "--no-bench",
            "--bundle",
        ],
    )
    assert res_dec.exit_code == 0, res_dec.output
    for fname in [
        "cli_ctrl.c",
        "cli_ctrl.h",
        "cli_ctrl.rs",
        "cli_ctrl.py",
        "cli_ctrl.ir.json",
        "cli_ctrl.mermaid.md",
        "cli_ctrl.certificate.json",
    ]:
        assert (out_dir / fname).is_file(), f"Missing {fname}"
    cert = json.loads((out_dir / "cli_ctrl.certificate.json").read_text(encoding="utf-8"))
    assert cert["model_name"] == "cli_ctrl"

    bundle_dir = out_dir / "cli_ctrl.gbundle"
    assert bundle_dir.is_dir()
    res_verify = runner.invoke(cli_app, ["verify", str(bundle_dir)])
    assert res_verify.exit_code == 0, res_verify.output
    assert "VERIFIED_REPRODUCED" in res_verify.output
