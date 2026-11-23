"""Tests for Z3 SMT formal verifier, adversarial falsifier, CEGIS loop, and certificates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from glassbox.cegis import decompile_with_cegis, run_decompile_pipeline
from glassbox.certificate import verify_certificate_integrity
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
        if abs(w) < 1e-12:
            continue
        term = BinaryExpr(op="mul", left=ConstExpr(value=w), right=VarExpr(index=idx, name=name))
        acc = term if acc is None else BinaryExpr(op="add", left=acc, right=term)
    if acc is None:
        return ConstExpr(value=bias)
    if abs(bias) < 1e-12:
        return acc
    return BinaryExpr(op="add", left=acc, right=ConstExpr(value=bias))


def _export_exact_relu_mlp(tmp_path: Path) -> tuple[LoadedModel, DecompiledProgram]:
    """Export f(x0, x1) = 2*relu(x0 - 0.5*x1) - relu(-x0) + 0.25*x1 + 1.0 and exact IR."""
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    with torch.no_grad():
        # Hidden units: h0 = relu(x0 - 0.5*x1), h1 = relu(-x0), h2 = relu(x1 + 5.0) = x1 + 5.0 on [-2,2]
        net[0].weight.copy_(torch.tensor([[1.0, -0.5], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        net[0].bias.copy_(torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32))
        # Output: 2*h0 - 1*h1 + 0.25*(x1 + 5.0) - 0.25 = 2*h0 - h1 + 0.25*x1 + 1.0
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

    # Exact 4-regime partition on guards g0: x0 - 0.5*x1 <= 0, g1: -x0 <= 0 (i.e. x0 >= 0)
    names = ["x0", "x1"]
    # g0 True (h0=0), g1 True (x0>=0 => h1=0): 0.25*x1 + 1.0
    leaf_tt = RegimeLeaf(
        regime_id=0,
        name="r0",
        description="h0=0, h1=0",
        output_exprs=[_linear_expr([0.0, 0.25], 1.0, names)],
    )
    # g0 True (h0=0), g1 False (x0<0 => h1=-x0): x0 + 0.25*x1 + 1.0
    leaf_tf = RegimeLeaf(
        regime_id=1,
        name="r1",
        description="h0=0, h1=-x0",
        output_exprs=[_linear_expr([1.0, 0.25], 1.0, names)],
    )
    # g0 False (h0=x0-0.5*x1), g1 True (x0>=0 => h1=0): 2*x0 - 0.75*x1 + 1.0
    leaf_ft = RegimeLeaf(
        regime_id=2,
        name="r2",
        description="h0>0, h1=0",
        output_exprs=[_linear_expr([2.0, -0.75], 1.0, names)],
    )
    # g0 False (h0=x0-0.5*x1), g1 False (x0<0 => h1=-x0): 3*x0 - 0.75*x1 + 1.0
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
        # h0 = relu(x1 - x0), h1 = relu(x0 - x1)
        net[0].weight.copy_(torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=torch.float32))
        net[0].bias.copy_(torch.tensor([0.0, 0.0], dtype=torch.float32))
        # logit0 = 2*h0, logit1 = 2*h1 -> class 0 iff x0 <= x1, class 1 iff x0 > x1
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
    torch.manual_seed(7)
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
    """Exact piecewise-linear ReLU MLP + matching IR yields genuine PROVEN_SMT (unsat)."""
    model, exact_prog = _export_exact_relu_mlp(tmp_path)
    cfg = DecompileConfig(tolerance_eps=1e-4, verify_smt=True, smt_timeout_s=5.0)

    res = verify_smt(model, exact_prog, cfg, iteration=0)
    assert res.checked is True
    assert res.proven is True
    assert res.timed_out is False
    assert res.counterexample is None
    assert res.reason == "unsat_proven"
    assert res.time_ms >= 0.0


def test_smt_finds_real_counterexample_on_perturbed_program(tmp_path: Path) -> None:
    """Perturbing one leaf constant produces a verified real z3_smt counterexample."""
    model, exact_prog = _export_exact_relu_mlp(tmp_path)
    perturbed = exact_prog.model_copy(deep=True)
    # Perturb leaf_ft constant bias by +0.30 (exceeds eps=0.05).
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
    assert res.timed_out is False
    assert res.counterexample is not None
    cex = res.counterexample
    assert cex.source == "z3_smt"
    assert cex.iteration == 1
    assert cex.max_abs_error > 0.05

    # Independently confirm numerical violation against real ONNXRuntime + IR evaluation.
    nn_y = model.predict(np.asarray([cex.x], dtype=np.float64))[0]
    prog_y, rid = perturbed.evaluate_one(cex.x)
    real_err = float(np.max(np.abs(nn_y - np.asarray(prog_y, dtype=np.float64))))
    assert real_err > 0.05
    assert abs(real_err - cex.max_abs_error) < 1e-6
    assert rid == cex.regime_id


def test_smt_classification_prove_and_falsify(tmp_path: Path) -> None:
    """SMT proves exact classification IR and falsifies flipped class_id with real margin."""
    model, exact_cls_prog = _export_relu_classifier(tmp_path)
    cfg = DecompileConfig(
        task_type=TaskType.CLASSIFICATION,
        classification_margin=0.05,
        verify_smt=True,
        smt_timeout_s=5.0,
    )

    proven_res = verify_smt(model, exact_cls_prog, cfg, iteration=0)
    assert proven_res.checked is True
    assert proven_res.proven is True
    assert proven_res.counterexample is None

    broken = exact_cls_prog.model_copy(deep=True)
    assert isinstance(broken.root, GuardNode)
    assert isinstance(broken.root.false_branch, RegimeLeaf)
    broken.root.false_branch.class_id = 0

    falsified = verify_smt(model, broken, cfg, iteration=2)
    assert falsified.checked is True
    assert falsified.proven is False
    assert falsified.counterexample is not None
    cex = falsified.counterexample
    assert cex.source == "z3_smt"
    assert cex.nn_class == 1
    assert cex.program_class == 0
    assert cex.nn_output[1] - cex.nn_output[0] >= cfg.classification_margin - 1e-6


def test_tanh_skips_smt_and_adversarial_falsifies(tmp_path: Path) -> None:
    """Non-piecewise-linear Tanh skips SMT cleanly while adversarial finds violations."""
    model = _export_tanh_controller(tmp_path)
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
    cfg = DecompileConfig(
        tolerance_eps=0.05,
        verify_smt=True,
        adversarial_restarts=12,
        seed=11,
    )

    smt_res = verify_smt(model, crude, cfg)
    assert smt_res.checked is False
    assert smt_res.proven is False
    assert smt_res.reason == "model_not_piecewise_linear"

    stats, cexs = evaluate_and_falsify(model, crude, cfg, iteration=0)
    assert stats.num_samples >= 4096
    assert stats.max_abs_error > 0.05
    assert len(cexs) > 0
    assert cexs[0].source == "adversarial"
    assert cexs[0].max_abs_error > 0.05
    assert stats.max_abs_error >= cexs[0].max_abs_error - 1e-9


def test_cegis_end_to_end_relu_and_certificate_integrity(tmp_path: Path) -> None:
    """End-to-end CEGIS on a ReLU controller achieves high accuracy/proof and valid certificate."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        # f(x0, x1) = 1.5*relu(x0) - 0.8*relu(-x0) + 0.0 -> 2 exact linear regimes split at x0=0
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
    onnx_path = tmp_path / "relu_ctrl.onnx"
    model = LoadedModel.from_pytorch(net, domain, onnx_path, name="relu_ctrl")

    seen_iters: list[CEGISIteration] = []
    cfg = DecompileConfig(
        max_regimes=6,
        max_depth=3,
        max_degree=1,
        tolerance_eps=0.05,
        max_cegis_iters=4,
        initial_samples=600,
        verify_smt=True,
        smt_timeout_s=4.0,
        adversarial_restarts=12,
        seed=42,
    )

    program, cert, cexs = decompile_with_cegis(model, cfg, on_iteration=seen_iters.append)

    assert len(seen_iters) >= 1
    assert len(cert.cegis_iterations) == len(seen_iters)
    assert cert.smt_checked is True
    assert cert.agreement_rate >= 0.98
    assert cert.max_abs_error <= 0.05 or cert.status == VerificationStatus.PROVEN_SMT
    assert verify_certificate_integrity(cert, onnx_path, program) is True

    # Tampering with program must break certificate integrity.
    tampered = program.model_copy(deep=True)
    tampered.name = "tampered_name"
    assert verify_certificate_integrity(cert, onnx_path, tampered) is False

    # Tampering with ONNX bytes must also break certificate integrity.
    onnx_path.write_bytes(onnx_path.read_bytes() + b"\x00")
    assert verify_certificate_integrity(cert, onnx_path, program) is False

    # Repaired counterexamples (if any were discovered early) must carry valid booleans.
    assert all(isinstance(c.repaired, bool) for c in cexs)


def test_cegis_end_to_end_tanh_and_pipeline_bundle(tmp_path: Path) -> None:
    """End-to-end CEGIS on a smooth Tanh controller improves fidelity and returns DecompileResult."""
    model = _export_tanh_controller(tmp_path)
    cfg = DecompileConfig(
        max_regimes=6,
        max_depth=3,
        max_degree=1,
        tolerance_eps=0.10,
        max_cegis_iters=3,
        initial_samples=600,
        verify_smt=True,
        adversarial_restarts=10,
        seed=42,
    )

    result = run_decompile_pipeline(model, cfg, run_id="tanh-test-run")
    assert result.run_id == "tanh-test-run"
    assert result.certificate.smt_checked is False
    assert result.certificate.smt_proven is False
    assert result.certificate.agreement_rate >= 0.90
    assert verify_certificate_integrity(result.certificate, model.metadata.path, result.program)
    # Final reported certificate error cannot be worse than initial iteration error.
    first_err = result.certificate.cegis_iterations[0].max_abs_error
    assert result.certificate.max_abs_error <= first_err + 1e-6
