"""Tests for the Glassbox benchmark model zoo and runtime benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx

from glassbox.benchmark import run_benchmarks
from glassbox.emitters import emit_all
from glassbox.ir import (
    BinaryExpr,
    ClampExpr,
    ConstExpr,
    DecompiledProgram,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    VarExpr,
)
from glassbox.loader import load_onnx
from glassbox.types import DomainSpec, TaskType
from glassbox.zoo import ZOO_SPECS, list_zoo_models, load_zoo_model_meta, train_zoo

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def test_zoo_models_present_and_valid() -> None:
    expected_ids = {
        "cartpole_policy",
        "drone_altitude_pid",
        "reactor_safety_guard",
        "nonlinear_actuator",
    }
    assert set(ZOO_SPECS.keys()) == expected_ids

    records = list_zoo_models(MODELS_DIR)
    assert len(records) == 4
    for rec in records:
        assert rec["available"] is True
        meta = load_zoo_model_meta(MODELS_DIR, rec["id"])
        domain = DomainSpec.model_validate(meta["domain"])
        onnx_path = MODELS_DIR / meta["onnx_file"]
        assert onnx_path.exists()

        proto = onnx.load(str(onnx_path))
        ops = [n.op_type for n in proto.graph.node]
        assert all(op in {"Gemm", "Relu", "Tanh"} for op in ops)

        loaded = load_onnx(onnx_path)
        assert loaded.metadata.domain == domain
        assert loaded.extract_mlp_layers() is not None
        expected_pwl = rec["id"] != "nonlinear_actuator"
        assert loaded.metadata.is_piecewise_linear is expected_pwl


def test_train_zoo_deterministic_and_idempotent(tmp_path: Path) -> None:
    paths_first = train_zoo(tmp_path, force=True)
    assert len(paths_first) == 4
    for p in paths_first:
        assert p.exists()
        assert p.with_suffix(".meta.json").exists()

    # Second call without force should reuse existing files.
    mtimes_before = [p.stat().st_mtime_ns for p in paths_first]
    paths_second = train_zoo(tmp_path, force=False)
    mtimes_after = [p.stat().st_mtime_ns for p in paths_second]
    assert mtimes_before == mtimes_after


def _build_sample_drone_program(domain: DomainSpec) -> DecompiledProgram:
    pd = BinaryExpr(
        op="add",
        left=ConstExpr(value=0.52),
        right=BinaryExpr(
            op="sub",
            left=BinaryExpr(
                op="mul",
                left=ConstExpr(value=0.18),
                right=VarExpr(index=0, name="alt_error"),
            ),
            right=BinaryExpr(
                op="mul",
                left=ConstExpr(value=0.14),
                right=VarExpr(index=1, name="vert_vel"),
            ),
        ),
    )
    low_batt = BinaryExpr(
        op="add",
        left=pd,
        right=BinaryExpr(
            op="mul",
            left=ConstExpr(value=0.06),
            right=BinaryExpr(
                op="sub",
                left=ConstExpr(value=11.8),
                right=VarExpr(index=2, name="battery_v"),
            ),
        ),
    )
    high_batt = BinaryExpr(
        op="sub",
        left=pd,
        right=BinaryExpr(
            op="mul",
            left=ConstExpr(value=0.02),
            right=BinaryExpr(
                op="sub",
                left=VarExpr(index=2, name="battery_v"),
                right=ConstExpr(value=11.8),
            ),
        ),
    )
    root = GuardNode(
        guard=LinearGuard(
            weights=[0.0, 0.0, 1.0],
            bias=-11.8,
            feature_names=domain.feature_names,
            op="<=",
        ),
        true_branch=RegimeLeaf(
            regime_id=0,
            name="low_battery_sag",
            description="battery_v <= 11.8",
            output_exprs=[ClampExpr(operand=low_batt, low=0.0, high=1.0)],
            sample_count=100,
        ),
        false_branch=RegimeLeaf(
            regime_id=1,
            name="nominal_battery",
            description="battery_v > 11.8",
            output_exprs=[ClampExpr(operand=high_batt, low=0.0, high=1.0)],
            sample_count=100,
        ),
    )
    return DecompiledProgram(
        name="drone_altitude_pid_sample",
        task_type=TaskType.REGRESSION,
        domain=domain,
        root=root,
        is_linear_ir=True,
    )


def test_run_benchmarks_all_runtimes(tmp_path: Path) -> None:
    loaded = load_onnx(MODELS_DIR / "drone_altitude_pid.onnx")
    program = _build_sample_drone_program(loaded.metadata.domain)

    rng = np.random.default_rng(7)
    X_gold = rng.uniform(
        loaded.metadata.domain.lower_bounds,
        loaded.metadata.domain.upper_bounds,
        size=(5, 3),
    )
    golden = [
        (row.tolist(), *program.evaluate_one(row.tolist()))
        for row in X_gold
    ]
    emitted = emit_all(program, golden_vectors=golden)

    entries = run_benchmarks(
        loaded,
        program,
        emitted,
        work_dir=tmp_path,
        n_warmup=20,
        n_iters=200,
    )
    by_name = {e.runtime: e for e in entries}
    assert set(by_name.keys()) == {
        "ONNX Runtime (CPU)",
        "PyTorch (Eager CPU)",
        "Decompiled Python (Zero-Dep)",
        "Decompiled C99 (Native DLL)",
    }
    for entry in entries:
        assert entry.available is True, f"{entry.runtime} unavailable: {entry.note}"
        assert entry.p50_us >= 0.0
        assert entry.throughput_ips > 0.0
        assert entry.artifact_bytes > 0
        assert entry.speedup_vs_onnx > 0.0

    assert by_name["PyTorch (Eager CPU)"].max_diff_vs_ref < 1e-5
    assert by_name["Decompiled Python (Zero-Dep)"].max_diff_vs_ref < 1e-9
    assert by_name["Decompiled C99 (Native DLL)"].max_diff_vs_ref < 1e-6


def test_run_benchmarks_handles_missing_emitted_code() -> None:
    loaded = load_onnx(MODELS_DIR / "cartpole_policy.onnx")
    domain = loaded.metadata.domain
    program = DecompiledProgram(
        name="cartpole_stub",
        task_type=TaskType.CLASSIFICATION,
        domain=domain,
        root=RegimeLeaf(
            regime_id=0,
            name="always_left",
            description="stub",
            output_exprs=[ConstExpr(value=1.0), ConstExpr(value=-1.0)],
            class_id=0,
            class_name="LEFT",
        ),
        is_linear_ir=True,
    )
    entries = run_benchmarks(loaded, program, {}, n_warmup=5, n_iters=20)
    by_name = {e.runtime: e for e in entries}
    assert by_name["ONNX Runtime (CPU)"].available is True
    assert by_name["Decompiled Python (Zero-Dep)"].available is False
    assert by_name["Decompiled C99 (Native DLL)"].available is False
