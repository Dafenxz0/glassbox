"""Unit tests for ONNX/PyTorch model loading and regime partitioning."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from glassbox.ir import GuardNode, RegimeLeaf
from glassbox.loader import DenseLayer, LoadedModel, from_pytorch, load_onnx
from glassbox.regimes import (
    RegimePartition,
    extract_regimes,
    refine_regime_around_counterexamples,
)
from glassbox.types import DecompileConfig, DomainSpec, TaskType


def _manual_mlp_forward(layers: list[DenseLayer], X: np.ndarray) -> np.ndarray:
    cur = np.asarray(X, dtype=np.float64)
    for layer in layers:
        z = cur @ layer.weight.T + layer.bias
        if layer.activation == "Relu":
            cur = np.maximum(0.0, z)
        elif layer.activation == "Tanh":
            cur = np.tanh(z)
        elif layer.activation == "Sigmoid":
            cur = 1.0 / (1.0 + np.exp(-z))
        else:
            cur = z
    return cur


def test_from_pytorch_and_load_onnx_relu_and_tanh(tmp_path: Path) -> None:
    torch.manual_seed(0)

    # 1. ReLU MLP regression
    relu_net = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    )
    relu_domain = DomainSpec(
        feature_names=["x0", "x1", "x2"],
        lower_bounds=[-2.0, -2.0, -2.0],
        upper_bounds=[2.0, 2.0, 2.0],
        output_names=["y0", "y1"],
        task_type=TaskType.REGRESSION,
    )
    relu_onnx = tmp_path / "relu_mlp.onnx"
    loaded_relu = LoadedModel.from_pytorch(relu_net, relu_domain, relu_onnx, name="relu_test")

    assert relu_onnx.is_file()
    assert relu_onnx.with_suffix(".meta.json").is_file()
    assert loaded_relu.metadata.name == "relu_test"
    assert loaded_relu.metadata.sha256 == hashlib.sha256(relu_onnx.read_bytes()).hexdigest()
    assert loaded_relu.metadata.size_bytes == relu_onnx.stat().st_size
    assert loaded_relu.metadata.is_piecewise_linear is True
    assert loaded_relu.metadata.activations == ["Relu", "Relu", "Identity"]
    assert loaded_relu.metadata.input_dim == 3
    assert loaded_relu.metadata.output_dim == 2

    layers = loaded_relu.extract_mlp_layers()
    assert layers is not None
    assert len(layers) == 3
    assert [layer.activation for layer in layers] == ["Relu", "Relu", "Identity"]

    rng = np.random.default_rng(123)
    X = rng.uniform(-2.0, 2.0, size=(64, 3))
    ort_out = loaded_relu.predict(X)
    assert ort_out.dtype == np.float64
    assert ort_out.shape == (64, 2)

    manual_out = _manual_mlp_forward(layers, X)
    assert np.max(np.abs(ort_out - manual_out)) < 1e-5

    hidden = loaded_relu.hidden_activations(X)
    assert len(hidden) == 2
    assert hidden[0].shape == (64, 8)
    assert hidden[1].shape == (64, 6)
    assert np.all(hidden[0] >= 0.0)
    assert np.all(hidden[1] >= 0.0)

    # Reload via load_onnx without domain -> picks up sibling .meta.json
    reloaded = load_onnx(relu_onnx)
    assert reloaded.metadata.domain == relu_domain
    assert np.allclose(reloaded.predict(X), ort_out)

    # 2. Tanh MLP -> NOT piecewise linear
    tanh_net = nn.Sequential(
        nn.Linear(2, 5),
        nn.Tanh(),
        nn.Linear(5, 1),
        nn.Sigmoid(),
    )
    tanh_domain = DomainSpec(
        feature_names=["a", "b"],
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        output_names=["prob"],
        task_type=TaskType.CLASSIFICATION,
        class_names=["neg", "pos"],
    )
    tanh_onnx = tmp_path / "tanh_mlp.onnx"
    loaded_tanh = from_pytorch(tanh_net, tanh_domain, tanh_onnx)
    assert loaded_tanh.metadata.is_piecewise_linear is False
    assert loaded_tanh.metadata.activations == ["Tanh", "Sigmoid"]

    tanh_layers = loaded_tanh.extract_mlp_layers()
    assert tanh_layers is not None
    X2 = rng.uniform(-1.0, 1.0, size=(40, 2))
    tanh_pred = loaded_tanh.predict(X2)
    assert np.max(np.abs(tanh_pred - _manual_mlp_forward(tanh_layers, X2))) < 1e-5

    cls_pred = loaded_tanh.predict_classes(X2)
    assert cls_pred.dtype == np.int64
    assert cls_pred.shape == (40,)
    assert np.array_equal(cls_pred, (tanh_pred[:, 0] > 0.5).astype(np.int64))


def test_load_onnx_matmul_add_clip_and_default_domain(tmp_path: Path) -> None:
    # Build a synthetic ONNX graph using Gemm (with alpha/beta/transB=0), MatMul+Add, and Clip(0, +inf)
    w1 = np.array([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]], dtype=np.float32)  # (in=3, out=2)
    b1 = np.array([0.2, -0.4], dtype=np.float32)
    w2 = np.array([[2.0, -1.0], [0.5, 1.5]], dtype=np.float32)  # (in=2, out=2)
    b2 = np.array([0.1, -0.3], dtype=np.float32)
    clip_min = np.array(0.0, dtype=np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gemm", ["x", "w1", "b1"], ["h1_pre"], alpha=0.5, beta=2.0, transB=0),
            helper.make_node("Clip", ["h1_pre", "clip_min"], ["h1"]),
            helper.make_node("MatMul", ["h1", "w2"], ["mm2"]),
            helper.make_node("Add", ["mm2", "b2"], ["y"]),
        ],
        name="matmul_clip_mlp",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])],
        initializer=[
            numpy_helper.from_array(w1, name="w1"),
            numpy_helper.from_array(b1, name="b1"),
            numpy_helper.from_array(w2, name="w2"),
            numpy_helper.from_array(b2, name="b2"),
            numpy_helper.from_array(clip_min, name="clip_min"),
        ],
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_path = tmp_path / "custom_no_meta.onnx"
    onnx.save(model_proto, str(onnx_path))

    loaded = load_onnx(onnx_path)
    assert loaded.metadata.domain.feature_names == ["x0", "x1", "x2"]
    assert loaded.metadata.domain.output_names == ["y0", "y1"]
    assert loaded.metadata.domain.task_type == TaskType.REGRESSION
    assert loaded.metadata.domain.lower_bounds == [-1.0, -1.0, -1.0]
    assert loaded.metadata.domain.upper_bounds == [1.0, 1.0, 1.0]
    assert loaded.metadata.is_piecewise_linear is True

    layers = loaded.extract_mlp_layers()
    assert layers is not None
    assert len(layers) == 2
    assert layers[0].activation == "Relu"
    assert layers[1].activation == "Identity"

    X = np.random.default_rng(7).uniform(-1.0, 1.0, size=(32, 3))
    assert np.max(np.abs(loaded.predict(X) - _manual_mlp_forward(layers, X))) < 1e-5


def test_non_mlp_skip_connection_falls_back_gracefully(tmp_path: Path) -> None:
    class SkipNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(3, 3)
            self.fc2 = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x)) + x
            return self.fc2(h)

    dom = DomainSpec(
        feature_names=["x0", "x1", "x2"],
        lower_bounds=[-1.0, -1.0, -1.0],
        upper_bounds=[1.0, 1.0, 1.0],
        output_names=["y0", "y1"],
        task_type=TaskType.REGRESSION,
    )
    loaded = from_pytorch(SkipNet(), dom, tmp_path / "skip.onnx")
    assert loaded.extract_mlp_layers() is None
    assert loaded.metadata.is_piecewise_linear is False
    assert loaded.hidden_activations(np.zeros((4, 3))) == []


def _check_partition_invariants(
    partition: RegimePartition, X: np.ndarray, max_depth: int | None = None
) -> None:
    leaves: list[RegimeLeaf] = partition.leaves()
    rids = [leaf.regime_id for leaf in leaves]
    assert rids == list(range(len(leaves)))
    assert set(partition.leaf_indices.keys()) == set(rids)

    assigned = partition.assign(X)
    assert assigned.dtype == np.int64
    assert assigned.shape == (len(X),)

    total_count = 0
    for rid, idx in partition.leaf_indices.items():
        assert np.all(assigned[idx] == rid)
        total_count += len(idx)
    assert total_count == len(X)

    def _walk_guards(node: GuardNode | RegimeLeaf, depth: int) -> int:
        if isinstance(node, RegimeLeaf):
            return depth
        assert len(node.guard.active_terms()) <= 3
        return max(
            _walk_guards(node.true_branch, depth + 1),
            _walk_guards(node.false_branch, depth + 1),
        )

    actual_depth = _walk_guards(partition.root, 0)
    if max_depth is not None:
        assert actual_depth <= max_depth


def test_oblique_tree_regression_and_classification(tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Piecewise-linear ReLU network on 4D input
    reg_net = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )
    reg_dom = DomainSpec(
        feature_names=["f0", "f1", "f2", "f3"],
        lower_bounds=[-1.0] * 4,
        upper_bounds=[1.0] * 4,
        output_names=["y0"],
        task_type=TaskType.REGRESSION,
    )
    reg_model = from_pytorch(reg_net, reg_dom, tmp_path / "reg.onnx")
    rng = np.random.default_rng(11)
    X_reg = rng.uniform(-1.0, 1.0, size=(300, 4))
    Y_reg = reg_model.predict(X_reg)

    cfg_reg = DecompileConfig(max_regimes=6, max_depth=3, regime_method="oblique_tree", seed=42)
    part_reg = extract_regimes(reg_model, X_reg, Y_reg, cfg_reg)
    assert 1 <= len(part_reg.leaves()) <= cfg_reg.max_regimes
    _check_partition_invariants(part_reg, X_reg, max_depth=cfg_reg.max_depth)

    # Classification network with diagonal decision boundary x0 + x1 > 0
    cls_net = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    )
    with torch.no_grad():
        cls_net[0].weight.zero_()
        cls_net[0].bias.zero_()
        cls_net[0].weight[0, 0] = 2.0
        cls_net[0].weight[0, 1] = 2.0
        cls_net[0].weight[1, 0] = -2.0
        cls_net[0].weight[1, 1] = -2.0
        cls_net[2].weight.zero_()
        cls_net[2].bias.zero_()
        cls_net[2].weight[0, 0] = 1.0
        cls_net[2].weight[1, 1] = 1.0

    cls_dom = DomainSpec(
        feature_names=["x0", "x1", "x2"],
        lower_bounds=[-1.0] * 3,
        upper_bounds=[1.0] * 3,
        output_names=["logit0", "logit1"],
        task_type=TaskType.CLASSIFICATION,
        class_names=["c0", "c1"],
    )
    cls_model = from_pytorch(cls_net, cls_dom, tmp_path / "cls.onnx")
    X_cls = rng.uniform(-1.0, 1.0, size=(240, 3))
    Y_cls = cls_model.predict(X_cls)

    cfg_cls = DecompileConfig(
        task_type=TaskType.CLASSIFICATION,
        max_regimes=4,
        max_depth=3,
        regime_method="oblique_tree",
        seed=42,
    )
    part_cls = extract_regimes(cls_model, X_cls, Y_cls, cfg_cls)
    assert 2 <= len(part_cls.leaves()) <= cfg_cls.max_regimes
    _check_partition_invariants(part_cls, X_cls, max_depth=cfg_cls.max_depth)
    # Impurity reduction should yield high leaf class purity on this linearly separable net
    y_true_cls = cls_model.predict_classes(X_cls)
    leaf_purities = []
    for leaf in part_cls.leaves():
        idx = part_cls.leaf_indices[leaf.regime_id]
        if len(idx) > 0:
            maj = int(np.bincount(y_true_cls[idx]).argmax())
            leaf_purities.append(float(np.mean(y_true_cls[idx] == maj)))
    assert np.mean(leaf_purities) > 0.90


def test_activation_clustering_and_counterexample_refinement(tmp_path: Path) -> None:
    torch.manual_seed(9)
    net = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )
    dom = DomainSpec(
        feature_names=["x0", "x1", "x2", "x3"],
        lower_bounds=[-1.5] * 4,
        upper_bounds=[1.5] * 4,
        output_names=["y"],
        task_type=TaskType.REGRESSION,
    )
    model = from_pytorch(net, dom, tmp_path / "act_net.onnx")
    rng = np.random.default_rng(99)
    X = rng.uniform(-1.5, 1.5, size=(250, 4))
    Y = model.predict(X)

    cfg_act = DecompileConfig(max_regimes=6, max_depth=4, regime_method="activation", seed=9)
    part_act = extract_regimes(model, X, Y, cfg_act)
    assert 2 <= len(part_act.leaves()) <= cfg_act.max_regimes
    _check_partition_invariants(part_act, X, max_depth=cfg_act.max_depth)

    # Counterexample refinement must split each failing leaf into 2 sub-regimes sequentially
    initial_leaves = len(part_act.leaves())
    ce_xs = X[:3]  # pick concrete points from existing dataset
    failing_rids = set(part_act.assign(ce_xs).tolist())

    refined = refine_regime_around_counterexamples(part_act, model, X, Y, ce_xs, cfg_act)
    assert len(refined.leaves()) == initial_leaves + len(failing_rids)
    _check_partition_invariants(refined, X)

    # Also test refining when counterexample_xs includes an unseen point not in X
    unseen_ce = np.array([[1.4, -1.4, 0.5, -0.5]], dtype=np.float64)
    refined2 = refine_regime_around_counterexamples(refined, model, X, Y, unseen_ce, cfg_act)
    assert len(refined2.leaves()) == len(refined.leaves()) + 1
    _check_partition_invariants(refined2, X)


def test_invalid_inputs_and_edge_cases(tmp_path: Path) -> None:
    net = nn.Sequential(nn.Linear(2, 2))
    dom = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        output_names=["y0", "y1"],
        task_type=TaskType.REGRESSION,
    )
    model = from_pytorch(net, dom, tmp_path / "lin.onnx")
    empty_X = np.empty((0, 2), dtype=np.float64)
    assert model.predict(empty_X).shape == (0, 2)
    assert model.predict_classes(empty_X).shape == (0,)
    assert model.hidden_activations(empty_X) == []

    with pytest.raises(ValueError):
        model.predict(np.array([1.0, 2.0]))

    cfg = DecompileConfig(regime_method="unknown_method")
    with pytest.raises(ValueError):
        extract_regimes(model, np.zeros((10, 2)), np.zeros((10, 2)), cfg)
