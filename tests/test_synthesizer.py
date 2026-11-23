"""Unit tests for Glassbox symbolic synthesizer."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch
from torch import nn

from glassbox.ir import (
    BinaryExpr,
    ConstExpr,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    VarExpr,
    eval_expr,
)
from glassbox.loader import LoadedModel
from glassbox.regimes import RegimePartition
from glassbox.synthesizer import linear_to_expr, sympy_to_expr, synthesize_program
from glassbox.types import DecompileConfig, DomainSpec, TaskType


def test_linear_to_expr_and_pruning() -> None:
    expr = linear_to_expr([2.0, 1.0, 1e-7], 0.0, ["x0", "x1", "x2"])
    assert eval_expr(expr, [3.0, 5.0, 99.0]) == pytest.approx(11.0)
    # 2*x0 + x1 -> mul(2, x0) + x1 = 5 nodes; x2 pruned cleanly
    assert isinstance(expr, BinaryExpr)
    assert expr.op == "add"
    assert isinstance(expr.right, VarExpr)

    expr_neg = linear_to_expr([-1.0, 0.0], 3.0, ["x0", "x1"])
    assert eval_expr(expr_neg, [2.0, 10.0]) == pytest.approx(1.0)
    assert isinstance(expr_neg, BinaryExpr)

    expr_zero = linear_to_expr([1e-8, -1e-8], 1e-9, ["x0", "x1"])
    assert isinstance(expr_zero, ConstExpr)
    assert expr_zero.value == 0.0


def test_sympy_to_expr_polynomial_and_ops() -> None:
    x0, x1 = sp.symbols("x0 x1")
    poly = x0**2 - sp.Float(0.5) * x1 + sp.Float(1e-10) * x0
    expr = sympy_to_expr(poly, ["x0", "x1"], zero_tol=1e-6)
    assert eval_expr(expr, [2.0, 4.0]) == pytest.approx(2.0)
    assert eval_expr(expr, [-1.5, -2.0]) == pytest.approx(3.25)

    relu_sym = sp.Max(0, x0) + sp.Abs(x1)
    relu_expr = sympy_to_expr(relu_sym, ["x0", "x1"])
    assert eval_expr(relu_expr, [-2.0, -3.0]) == pytest.approx(3.0)
    assert eval_expr(relu_expr, [2.5, -1.0]) == pytest.approx(3.5)

    with pytest.raises(ValueError):
        sympy_to_expr(sp.Symbol("unknown_var"), ["x0", "x1"])


class _PiecewiseLinearNet(nn.Module):
    """Exact representation of: x0 <= 0 ? 2*x0 + x1 : -x0 + 3."""

    def __init__(self) -> None:
        super().__init__()
        # ReLU(-x0) = max(0, -x0), ReLU(x0) = max(0, x0)
        # x1 = ReLU(x1 + 10) - 10 on [-2, 2]
        # y = 2*x0 + x1 + 3*ReLU(x0) + 3*(1 - ?) -> easier:
        # Note on x0 <= 0: -x0 = ReLU(-x0), x0 = -ReLU(-x0)
        # On x0 > 0: x0 = ReLU(x0)
        # Wait: x0 <= 0 -> 2*x0 + x1 = -2*ReLU(-x0) + x1
        # x0 > 0 -> -x0 + 3 -> jump at 0, so test both partition recovery & continuous ReLU net.
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],   # h0 = ReLU(x0)
                        [-1.0, 0.0],  # h1 = ReLU(-x0)
                        [0.0, 1.0],   # h2 = ReLU(x1 + 5)
                        [0.0, -1.0],  # h3 = ReLU(-x1 + 5)
                    ]
                )
            )
            self.fc1.bias.copy_(torch.tensor([0.0, 0.0, 5.0, 5.0]))
            # Continuous PWL: x0 <= 0 ? 2*x0 + x1 : -x0 + x1
            self.fc2.weight.copy_(torch.tensor([[-1.0, -2.0, 0.5, -0.5]]))
            self.fc2.bias.copy_(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _QuadraticNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ClassifierNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_piecewise_linear_two_regime_exact_recovery(tmp_path) -> None:
    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["y0"],
        task_type=TaskType.REGRESSION,
    )
    model = LoadedModel.from_pytorch(_PiecewiseLinearNet(), domain, tmp_path / "pwl.onnx")

    rng = np.random.default_rng(42)
    X = rng.uniform([-2.0, -2.0], [2.0, 2.0], size=(400, 2))
    # Exact target: x0 <= 0 ? 2*x0 + x1 : -x0 + 3
    Y = np.where(X[:, 0] <= 0.0, 2.0 * X[:, 0] + X[:, 1], -X[:, 0] + 3.0)[:, None]

    left_idx = np.flatnonzero(X[:, 0] <= 0.0)
    right_idx = np.flatnonzero(X[:, 0] > 0.0)
    root = GuardNode(
        guard=LinearGuard(weights=[1.0, 0.0], bias=0.0, feature_names=["x0", "x1"], op="<="),
        true_branch=RegimeLeaf(
            regime_id=0, name="r0", description="", output_exprs=[ConstExpr(value=0.0)]
        ),
        false_branch=RegimeLeaf(
            regime_id=1, name="r1", description="", output_exprs=[ConstExpr(value=0.0)]
        ),
    )
    part = RegimePartition(root=root, leaf_indices={0: left_idx, 1: right_idx})

    cfg = DecompileConfig(max_degree=1, max_regimes=4)
    prog = synthesize_program(model, X, Y, cfg, partition=part)

    assert prog.is_piecewise_linear() is True
    assert prog.is_linear_ir is True
    preds = prog.predict(X)
    assert np.max(np.abs(preds - Y)) < 1e-8
    assert prog.ast_node_count() <= 16
    leaves = prog.leaves()
    assert len(leaves) == 2
    assert leaves[0].name == "regime_0_x0_lo"
    assert leaves[1].name == "regime_1_x0_hi"
    assert "samples=" in leaves[0].description
    assert "local_err=" in leaves[0].description


def test_piecewise_linear_end_to_end_with_extract_regimes(tmp_path) -> None:
    from glassbox.loader import LoadedModel

    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["y0"],
        task_type=TaskType.REGRESSION,
    )
    model = LoadedModel.from_pytorch(_PiecewiseLinearNet(), domain, tmp_path / "pwl_e2e.onnx")
    rng = np.random.default_rng(7)
    X = rng.uniform([-2.0, -2.0], [2.0, 2.0], size=(600, 2))
    Y = model.predict(X)

    cfg = DecompileConfig(max_degree=1, max_regimes=6, max_depth=3)
    prog = synthesize_program(model, X, Y, cfg)

    assert prog.is_piecewise_linear() is True
    preds = prog.predict(X)
    assert np.mean(np.abs(preds - Y)) < 1e-2
    assert prog.ast_node_count() < 120


def test_quadratic_degree2_recovery(tmp_path) -> None:
    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["y0"],
        task_type=TaskType.REGRESSION,
    )
    model = LoadedModel.from_pytorch(_QuadraticNet(), domain, tmp_path / "quad.onnx")

    rng = np.random.default_rng(123)
    X = rng.uniform([-2.0, -2.0], [2.0, 2.0], size=(300, 2))
    Y = (X[:, 0] ** 2 - 0.5 * X[:, 1])[:, None]

    root = RegimeLeaf(
        regime_id=0, name="r0", description="", output_exprs=[ConstExpr(value=0.0)]
    )
    part = RegimePartition(root=root, leaf_indices={0: np.arange(len(X))})

    cfg_deg2 = DecompileConfig(max_degree=2, max_regimes=4)
    prog = synthesize_program(model, X, Y, cfg_deg2, partition=part)

    assert prog.is_piecewise_linear() is False
    assert prog.is_linear_ir is False
    preds = prog.predict(X)
    assert np.max(np.abs(preds - Y)) < 1e-8
    assert prog.ast_node_count() <= 12

    # Ensure linear data under max_degree=2 does NOT gratuitously keep quadratic terms
    Y_lin = (2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0)[:, None]
    prog_lin = synthesize_program(model, X, Y_lin, cfg_deg2, partition=part)
    assert prog_lin.is_piecewise_linear() is True
    assert prog_lin.is_linear_ir is True
    assert np.max(np.abs(prog_lin.predict(X) - Y_lin)) < 1e-8


def test_multiclass_classification_synthesis(tmp_path) -> None:
    domain = DomainSpec(
        feature_names=["core_temp", "pressure"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["logit_0", "logit_1", "logit_2"],
        task_type=TaskType.CLASSIFICATION,
        class_names=["idle", "nominal", "alert"],
    )
    model = LoadedModel.from_pytorch(_ClassifierNet(), domain, tmp_path / "cls.onnx")

    rng = np.random.default_rng(99)
    X = rng.uniform([-2.0, -2.0], [2.0, 2.0], size=(450, 2))
    # Construct 3-class linearly separable scores across 2 regimes:
    # Regime 0 (core_temp <= 0): pure class 0 ("idle")
    # Regime 1 (core_temp > 0): mixed class 1 vs class 2 separated by pressure
    Y = np.zeros((len(X), 3), dtype=np.float64)
    mask0 = X[:, 0] <= 0.0
    Y[mask0, 0] = 3.0
    Y[mask0, 1] = -1.0
    Y[mask0, 2] = -1.0
    Y[~mask0, 0] = -2.0
    Y[~mask0, 1] = 2.0 * X[~mask0, 1]
    Y[~mask0, 2] = -2.0 * X[~mask0, 1]

    idx0 = np.flatnonzero(mask0)
    idx1 = np.flatnonzero(~mask0)
    root = GuardNode(
        guard=LinearGuard(
            weights=[1.0, 0.0],
            bias=0.0,
            feature_names=["core_temp", "pressure"],
            op="<=",
        ),
        true_branch=RegimeLeaf(
            regime_id=0, name="r0", description="", output_exprs=[ConstExpr(value=0.0)] * 3
        ),
        false_branch=RegimeLeaf(
            regime_id=1, name="r1", description="", output_exprs=[ConstExpr(value=0.0)] * 3
        ),
    )
    part = RegimePartition(root=root, leaf_indices={0: idx0, 1: idx1})

    cfg = DecompileConfig(task_type=TaskType.CLASSIFICATION, max_degree=1, max_regimes=4)
    prog = synthesize_program(model, X, Y, cfg, partition=part)

    assert prog.is_piecewise_linear() is True
    leaves = prog.leaves()
    assert len(leaves) == 2
    assert all(len(leaf.output_exprs) == 3 for leaf in leaves)
    # Pure leaf sets class_id; impure leaf leaves class_id=None and uses argmax(output_exprs)
    assert leaves[0].class_id == 0
    assert leaves[0].class_name == "idle"
    assert leaves[1].class_id is None

    true_cls = np.argmax(Y, axis=1)
    pred_cls = prog.predict_classes(X)
    acc = float(np.mean(pred_cls == true_cls))
    assert acc == pytest.approx(1.0)
    assert np.max(np.abs(prog.predict(X) - Y)) < 1e-8

    # Also check end-to-end extract_regimes on the PyTorch classifier
    Y_nn = model.predict(X)
    prog_e2e = synthesize_program(model, X, Y_nn, DecompileConfig(max_regimes=8, max_depth=4))
    assert prog_e2e.is_piecewise_linear() is True
    e2e_acc = float(np.mean(prog_e2e.predict_classes(X) == model.predict_classes(X)))
    assert e2e_acc >= 0.90


def test_weighted_synthesis_prioritizes_high_weight_points(tmp_path) -> None:
    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -2.0],
        upper_bounds=[2.0, 2.0],
        output_names=["y0"],
        task_type=TaskType.REGRESSION,
    )
    model = LoadedModel.from_pytorch(_PiecewiseLinearNet(), domain, tmp_path / "weighted.onnx")
    rng = np.random.default_rng(11)
    X = rng.uniform([-2.0, -2.0], [2.0, 2.0], size=(100, 2))
    # First 50 points follow y = 2*x0 + x1; last 50 have corrupted y unless weighted out
    Y = (2.0 * X[:, 0] + X[:, 1])[:, None]
    Y[50:] += 5.0
    weights = np.ones(100, dtype=np.float64)
    weights[:50] = 1e6
    weights[50:] = 1e-6

    part = RegimePartition(
        root=RegimeLeaf(
            regime_id=0, name="r0", description="", output_exprs=[ConstExpr(value=0.0)]
        ),
        leaf_indices={0: np.arange(100)},
    )
    prog = synthesize_program(model, X, Y, DecompileConfig(max_degree=1), weights=weights, partition=part)
    assert np.max(np.abs(prog.predict(X[:50]) - Y[:50])) < 1e-4
