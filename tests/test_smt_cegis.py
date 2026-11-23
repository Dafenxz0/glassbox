"""Tests for exact rational Z3 SMT verifier, adversarial falsifier, CEGIS, and bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from glassbox.cegis import decompile_with_cegis, run_decompile_pipeline, search_pareto_frontier
from glassbox.certificate import (
    compute_file_sha256,
    verify_artifact_bundle,
    verify_certificate_integrity,
    write_artifact_bundle,
)
from glassbox.ir import (
    BinaryExpr,
    ConstExpr,
    DecompiledProgram,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    VarExpr,
)
from glassbox.loader import LoadedModel
from glassbox.types import (
    ArtifactManifest,
    CEGISIteration,
    DecompileConfig,
    DomainSpec,
    TaskType,
    VerificationStatus,
)
from glassbox.verifier import evaluate_and_falsify, verify_smt


def _linear_expr(weights: list[float], bias: float, names: list[str]) -> BinaryExpr | ConstExpr:
    """Build a clean affine IR expression sum_i w_i * x_i + b."""
    acc: BinaryExpr | ConstExpr | None = None
    for idx, (w, name) in enumerate(zip(weights, names)):
        if w == 0.0:
            continue
        term = BinaryExpr(op="mul", left=ConstExpr(value=w), right=VarExpr(index=idx, name=name))
        acc = term if acc is None else BinaryExpr(op="add", left=acc, right=term)
    if acc is None:
        return ConstExpr(value=bias)
    if bias == 0.0:
        return acc
    return BinaryExpr(op="add", left=acc, right=ConstExpr(value=bias))


def _export_exact_relu_mlp(tmp_path: Path) -> tuple[LoadedModel, DecompiledProgram]:
    """Export f(x0, x1) = 2*relu(x0 - 0.5*x1) - relu(-x0) + 0.25*x1 + 1.0 and exact IR."""
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    with torch.no_grad():
        net[0].weight.copy_(
            torch.tensor([[1.0, -0.5], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        )
        net[0].bias.copy_(torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32))
        net[2].weight.copy_(torch.tensor([[2.0, -1.0, 0.25]], dtype=torch.float32))
        net[2].bias.copy_(torch.tensor([-0.25], dtype=torch.float32))

    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["u"],
        task_type=TaskType.REGRESSION,
    )
    model = LoadedModel.from_pytorch(net, domain, tmp_path / "exact_relu.onnx", name="exact_relu")

    names = ["x0", "x1"]
    leaf_tt = RegimeLeaf(
        regime_id=0,
        name="r0",
        description="h0=0, h1=0",
        output_exprs=[_linear_expr([0.0, 0.25], 1.0, names)],
    )
    leaf_tf = RegimeLeaf(
        regime_id=1,
        name="r1",
        description="h0=0, h1=-x0",
        output_exprs=[_linear_expr([1.0, 0.25], 1.0, names)],
    )
    leaf_ft = RegimeLeaf(
        regime_id=2,
        name="r2",
        description="h0>0, h1=0",
        output_exprs=[_linear_expr([2.0, -0.75], 1.0, names)],
    )
    leaf_ff = RegimeLeaf(
        regime_id=3,
        name="r3",
        description="h0>0, h1=-x0",
        output_exprs=[_linear_expr([3.0, -0.75], 1.0, names)],
    )

    g_x0_nonneg = LinearGuard(weights=[-1.0, 0.0], bias=0.0, feature_names=names, op="<=")
    root = GuardNode(
        guard=LinearGuard(weights=[1.0, -0.5], bias=0.0, feature_names=names, op="<="),
        true_branch=GuardNode(guard=g_x0_nonneg, true_branch=leaf_tt, false_branch=leaf_tf),
        false_branch=GuardNode(guard=g_x0_nonneg, true_branch=leaf_ft, false_branch=leaf_ff),
    )
    program = DecompiledProgram(
        name="exact_relu_prog",
        task_type=TaskType.REGRESSION,
        domain=domain,
        root=root,
        is_linear_ir=True,
    )
    return model, program


def _export_relu_classifier(tmp_path: Path) -> tuple[LoadedModel, DecompiledProgram]:
    """Export 2-class ReLU MLP whose exact class decision boundary is x0 - x1 <= 0."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=torch.float32))
        net[0].bias.copy_(torch.tensor([0.0, 0.0], dtype=torch.float32))
        net[2].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32))
        net[2].bias.copy_(torch.tensor([0.0, 0.0], dtype=torch.float32))

    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["c0", "c1"],
        task_type=TaskType.CLASSIFICATION,
        class_names=["safe", "alert"],
    )
    model = LoadedModel.from_pytorch(net, domain, tmp_path / "relu_cls.onnx", name="relu_cls")

    names = ["x0", "x1"]
    root = GuardNode(
        guard=LinearGuard(weights=[1.0, -1.0], bias=0.0, feature_names=names, op="<="),
        true_branch=RegimeLeaf(
            regime_id=0,
            name="safe_zone",
            description="x0 <= x1",
            output_exprs=[_linear_expr([-2.0, 2.0], 0.0, names), ConstExpr(value=0.0)],
            class_id=0,
            class_name="safe",
        ),
        false_branch=RegimeLeaf(
            regime_id=1,
            name="alert_zone",
            description="x0 > x1",
            output_exprs=[ConstExpr(value=0.0), _linear_expr([2.0, -2.0], 0.0, names)],
            class_id=1,
            class_name="alert",
        ),
    )
    program = DecompiledProgram(
        name="relu_cls_prog",
        task_type=TaskType.CLASSIFICATION,
        domain=domain,
        root=root,
        is_linear_ir=True,
    )
    return model, program


def _export_tanh_controller(tmp_path: Path) -> LoadedModel:
    """Export a smooth Tanh controller MLP."""
    net = nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 1))
    with torch.no_grad():
        net[0].weight.copy_(
            torch.tensor(
                [
                    [0.8, -0.3],
                    [-0.4, 0.7],
                    [0.5, 0.5],
                    [-0.6, -0.2],
                    [0.2, -0.8],
                    [0.3, 0.4],
                ],
                dtype=torch.float32,
            )
        )
        net[0].bias.copy_(torch.tensor([0.1, -0.1, 0.0, 0.2, -0.2, 0.05], dtype=torch.float32))
        net[2].weight.copy_(torch.tensor([[0.6, -0.5, 0.4, -0.3, 0.5, 0.2]], dtype=torch.float32))
        net[2].bias.copy_(torch.tensor([0.1], dtype=torch.float32))

    domain = DomainSpec(
        feature_names=["pos", "vel"],
        lower_bounds=[-1.5, -1.5],
        upper_bounds=[1.5, 1.5],
        output_names=["thrust"],
        task_type=TaskType.REGRESSION,
    )
    return LoadedModel.from_pytorch(net, domain, tmp_path / "tanh_ctrl.onnx", name="tanh_ctrl")


def test_smt_proves_exact_relu_mlp_unsat(tmp_path: Path) -> None:
    """Exact piecewise-linear ReLU MLP + matching IR yields PROVEN_REAL_GLOBAL (unsat)."""
    model, exact_prog = _export_exact_relu_mlp(tmp_path)
    cfg = DecompileConfig(tolerance_eps=1e-6, verify_smt=True, smt_timeout_s=5.0)

    res = verify_smt(model, exact_prog, cfg, iteration=0)
    assert res.checked is True
    assert res.proven is True
    assert res.timed_out is False
    assert res.status == VerificationStatus.PROVEN_REAL_GLOBAL
    assert res.counterexample is None
    assert res.reason == "unsat_proven"
    assert res.proof is not None
    assert res.proof.exact_rationals is True
    assert res.proof.sound_bounds_only is True
    assert res.proof.scope == "global"
    assert res.proof.solver_result == "unsat"
    assert res.proof.leaves_proven == 4
    assert res.proof.leaves_total == 4
    assert all(res.proof.leaf_proven_map.values())


def test_smt_finds_real_counterexample_on_perturbed_program(tmp_path: Path) -> None:
    """Perturbing one leaf constant produces a verified real z3_smt counterexample."""
    model, exact_prog = _export_exact_relu_mlp(tmp_path)
    perturbed = exact_prog.model_copy(deep=True)
    assert isinstance(perturbed.root, GuardNode)
    assert isinstance(perturbed.root.false_branch, GuardNode)
    assert isinstance(perturbed.root.false_branch.true_branch, RegimeLeaf)
    perturbed.root.false_branch.true_branch.output_exprs = [
        _linear_expr([2.0, -0.75], 1.30, ["x0", "x1"])
    ]

    cfg = DecompileConfig(tolerance_eps=0.05, verify_smt=True, smt_timeout_s=5.0)
    res = verify_smt(model, perturbed, cfg, iteration=1)

    assert res.checked is True
    assert res.proven is False
    assert res.status == VerificationStatus.COUNTEREXAMPLE_FOUND
    assert res.counterexample is not None
    cex = res.counterexample
    assert cex.source == "z3_smt"
    assert cex.iteration == 1
    assert cex.max_abs_error > 0.05

    nn_y = model.predict(np.asarray([cex.x], dtype=np.float64))[0]
    prog_y, rid = perturbed.evaluate_one(cex.x)
    real_err = float(np.max(np.abs(nn_y - np.asarray(prog_y, dtype=np.float64))))
    assert real_err > 0.05
    assert abs(real_err - cex.max_abs_error) < 1e-5
    assert rid == cex.regime_id


def test_smt_catches_tiny_1e4_perturbation_near_decision_boundary(tmp_path: Path) -> None:
    """Exact rational Z3 catches a 1e-4 guard shift and a 1e-4 leaf perturbation."""
    # 1. Classification: shift decision boundary x0 - x1 <= 0 by 1e-4 -> x0 - x1 - 1e-4 <= 0.
    cls_model, exact_cls_prog = _export_relu_classifier(tmp_path)
    shifted_cls = exact_cls_prog.model_copy(deep=True)
    assert isinstance(shifted_cls.root, GuardNode)
    shifted_cls.root.guard.bias = -1e-4

    cls_cfg = DecompileConfig(
        task_type=TaskType.CLASSIFICATION,
        classification_margin=0.0,
        verify_smt=True,
        smt_timeout_s=5.0,
    )
    cls_res = verify_smt(cls_model, shifted_cls, cls_cfg, iteration=0)
    assert cls_res.checked is True
    assert cls_res.proven is False
    assert cls_res.status == VerificationStatus.COUNTEREXAMPLE_FOUND
    assert cls_res.counterexample is not None
    assert cls_res.counterexample.source == "z3_smt"

    # 2. Regression: shift guard x0 - 0.5*x1 <= 0 by 1e-4 under tolerance_eps = 5e-5.
    reg_model, exact_reg_prog = _export_exact_relu_mlp(tmp_path)
    shifted_reg = exact_reg_prog.model_copy(deep=True)
    assert isinstance(shifted_reg.root, GuardNode)
    shifted_reg.root.guard.bias = -1e-4

    reg_cfg = DecompileConfig(tolerance_eps=5e-5, verify_smt=True, smt_timeout_s=5.0)
    reg_res = verify_smt(reg_model, shifted_reg, reg_cfg, iteration=0)
    assert reg_res.checked is True
    assert reg_res.proven is False
    assert reg_res.status == VerificationStatus.COUNTEREXAMPLE_FOUND
    assert reg_res.counterexample is not None
    assert reg_res.counterexample.max_abs_error > 5e-5


def test_smt_classification_global_vs_margin_proofs(tmp_path: Path) -> None:
    """classification_margin=0.0 yields PROVEN_REAL_GLOBAL; margin=0.05 yields PROVEN_REAL_MARGIN."""
    model, exact_cls_prog = _export_relu_classifier(tmp_path)

    global_cfg = DecompileConfig(
        task_type=TaskType.CLASSIFICATION,
        classification_margin=0.0,
        verify_smt=True,
        smt_timeout_s=5.0,
    )
    global_res = verify_smt(model, exact_cls_prog, global_cfg)
    assert global_res.proven is True
    assert global_res.status == VerificationStatus.PROVEN_REAL_GLOBAL
    assert global_res.proof is not None
    assert global_res.proof.scope == "global"

    margin_cfg = DecompileConfig(
        task_type=TaskType.CLASSIFICATION,
        classification_margin=0.05,
        verify_smt=True,
        smt_timeout_s=5.0,
    )
    margin_res = verify_smt(model, exact_cls_prog, margin_cfg)
    assert margin_res.proven is True
    assert margin_res.status == VerificationStatus.PROVEN_REAL_MARGIN
    assert margin_res.proof is not None
    assert margin_res.proof.scope == "margin_conditioned"


def test_artifact_bundle_reproduction_and_tamper_detection(tmp_path: Path) -> None:
    """write_artifact_bundle + verify_artifact_bundle reproduce proof and detect tampering/forgery."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32))
        net[0].bias.copy_(torch.tensor([0.0, 0.0], dtype=torch.float32))
        net[2].weight.copy_(torch.tensor([[1.5, -0.8]], dtype=torch.float32))
        net[2].bias.copy_(torch.tensor([0.2], dtype=torch.float32))

    domain = DomainSpec(
        feature_names=["err", "derr"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["cmd"],
        task_type=TaskType.REGRESSION,
    )
    onnx_path = tmp_path / "relu_bundle.onnx"
    model = LoadedModel.from_pytorch(net, domain, onnx_path, name="relu_bundle")
    cfg = DecompileConfig(
        max_regimes=4,
        max_depth=3,
        tolerance_eps=0.05,
        max_cegis_iters=3,
        initial_samples=400,
        verify_smt=True,
        smt_timeout_s=4.0,
        seed=42,
    )

    result = run_decompile_pipeline(model, cfg, run_id="bundle-run")
    assert result.certificate.status == VerificationStatus.PROVEN_REAL_GLOBAL
    assert len(result.sample_explanations) == 5
    assert len(result.regime_neuron_profiles) == len(result.program.leaves())

    bundle_dir = write_artifact_bundle(result, onnx_path, tmp_path / "test.gbundle")
    report = verify_artifact_bundle(bundle_dir, recompute_smt=True)
    assert report.verdict == "VERIFIED_REPRODUCED"
    assert report.hashes_valid is True
    assert report.smt_replayed is True
    assert report.replayed_status == VerificationStatus.PROVEN_REAL_GLOBAL
    assert report.numerical_checks_passed is True

    # 1. Tamper with program.ir.json without updating manifest -> HASH_MISMATCH.
    ir_file = bundle_dir / "program.ir.json"
    orig_ir_text = ir_file.read_text(encoding="utf-8")
    ir_file.write_text(orig_ir_text + "\n", encoding="utf-8")
    tamper_report = verify_artifact_bundle(bundle_dir, recompute_smt=True)
    assert tamper_report.verdict == "HASH_MISMATCH"
    assert tamper_report.hashes_valid is False

    # Restore and test forged PROVEN_REAL_GLOBAL certificate on a flawed program -> FALSIFIED_BY_REPLAY.
    flawed_prog = DecompiledProgram(
        name="forged_zero",
        task_type=TaskType.REGRESSION,
        domain=domain,
        root=RegimeLeaf(
            regime_id=0,
            name="zero",
            description="forged",
            output_exprs=[ConstExpr(value=0.0)],
        ),
        is_linear_ir=True,
    )
    forged_cert = result.certificate.model_copy(
        update={
            "program_sha256": flawed_prog.sha256(),
            "status": VerificationStatus.PROVEN_REAL_GLOBAL,
            "smt_checked": True,
            "smt_proven": True,
        }
    )
    forged_result = result.model_copy(
        update={"program": flawed_prog, "certificate": forged_cert, "code": {}}
    )
    forged_dir = write_artifact_bundle(forged_result, onnx_path, tmp_path / "forged.gbundle")
    # Update manifest hashes so hashes_valid=True, forcing replay to catch the lie.
    manifest = ArtifactManifest.model_validate_json(
        (forged_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest.files = {rel: compute_file_sha256(forged_dir / rel) for rel in manifest.files}
    (forged_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    forged_report = verify_artifact_bundle(forged_dir, recompute_smt=True)
    assert forged_report.hashes_valid is True
    assert forged_report.verdict == "FALSIFIED_BY_REPLAY"


def test_tanh_skips_smt_and_pareto_frontier(tmp_path: Path) -> None:
    """Non-piecewise-linear Tanh skips SMT cleanly, and search_pareto_frontier marks optimal front."""
    model = _export_tanh_controller(tmp_path)
    cfg = DecompileConfig(
        max_regimes=4,
        max_depth=3,
        tolerance_eps=0.10,
        max_cegis_iters=2,
        initial_samples=400,
        verify_smt=True,
        adversarial_restarts=8,
        seed=42,
    )

    seen_iters: list[CEGISIteration] = []
    prog, cert, _ = decompile_with_cegis(model, cfg, on_iteration=seen_iters.append)
    assert cert.smt_checked is False
    assert cert.smt_proven is False
    assert verify_certificate_integrity(cert, model.metadata.path, prog)

    frontier = search_pareto_frontier(model, cfg, budgets=(2, 4), run_c_bench=False)
    assert len(frontier) == 2
    assert any(c.is_pareto_optimal for c in frontier)
    assert all(c.code_c and c.code_rust and c.code_python for c in frontier)

    crude = DecompiledProgram(
        name="crude_zero",
        task_type=TaskType.REGRESSION,
        domain=model.metadata.domain,
        root=RegimeLeaf(
            regime_id=0,
            name="const_zero",
            description="zero",
            output_exprs=[ConstExpr(value=0.0)],
        ),
        is_linear_ir=True,
    )
    stats, cexs = evaluate_and_falsify(model, crude, cfg, iteration=0)
    assert stats.num_samples >= 4096
    assert len(cexs) > 0
