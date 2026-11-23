"""ONNX and PyTorch model loading, metadata extraction, and exact MLP parsing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

from glassbox.ir import (
    BinaryExpr,
    ClampExpr,
    ConstExpr,
    DecompiledProgram,
    Expr,
    UnaryExpr,
    VarExpr,
)
from glassbox.types import (
    DecisionExplanation,
    DomainSpec,
    LayerInfo,
    ModelMetadata,
    TaskType,
)

VALID_ACTIVATIONS = frozenset({"Relu", "Tanh", "Sigmoid", "Identity"})


@dataclass
class DenseLayer:
    """Extracted feed-forward dense layer with affine parameters and activation."""

    name: str
    weight: np.ndarray  # shape (out_features, in_features)
    bias: np.ndarray  # shape (out_features,)
    activation: str  # one of {"Relu", "Tanh", "Sigmoid", "Identity"}

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        if self.weight.ndim != 2:
            raise ValueError(f"DenseLayer weight must be 2D, got shape {self.weight.shape}")
        if self.bias.ndim != 1 or self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError(
                f"DenseLayer bias must have shape ({self.weight.shape[0]},), "
                f"got {self.bias.shape}"
            )
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported DenseLayer activation {self.activation!r}; "
                f"expected one of {sorted(VALID_ACTIVATIONS)}"
            )


def _apply_activation(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "Relu":
        return np.maximum(0.0, z)
    if activation == "Tanh":
        return np.tanh(z)
    if activation == "Sigmoid":
        return np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    if activation == "Identity":
        return z
    raise ValueError(f"Unsupported activation: {activation}")


def _extract_constant_from_node(node: onnx.NodeProto) -> np.ndarray | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
        if attr.name == "value_float":
            return np.array(attr.f, dtype=np.float32)
        if attr.name == "value_floats":
            return np.array(list(attr.floats), dtype=np.float32)
        if attr.name == "value_int":
            return np.array(attr.i, dtype=np.int64)
        if attr.name == "value_ints":
            return np.array(list(attr.ints), dtype=np.int64)
    return None


def _get_attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == onnx.AttributeProto.INT:
                return int(attr.i)
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
            if attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8")
    return default


def _broadcast_bias(raw_bias: np.ndarray, out_features: int) -> np.ndarray | None:
    b = np.asarray(raw_bias, dtype=np.float64)
    if b.ndim == 0:
        return np.full((out_features,), float(b), dtype=np.float64)
    if b.ndim == 1:
        if b.shape[0] == out_features:
            return b.copy()
        if b.shape[0] == 1:
            return np.full((out_features,), float(b[0]), dtype=np.float64)
        return None
    if b.ndim == 2 and b.shape[0] == 1:
        if b.shape[1] == out_features:
            return b.reshape(-1).copy()
        if b.shape[1] == 1:
            return np.full((out_features,), float(b[0, 0]), dtype=np.float64)
    return None


def _clip_is_relu(node: onnx.NodeProto, constants: dict[str, np.ndarray]) -> bool:
    min_val: float | None = None
    max_val: float | None = None

    if len(node.input) >= 2 and node.input[1]:
        if node.input[1] not in constants:
            return False
        min_arr = np.asarray(constants[node.input[1]], dtype=np.float64)
        if min_arr.size != 1:
            return False
        min_val = float(min_arr.reshape(()))
    else:
        attr_min = _get_attr(node, "min", None)
        if attr_min is not None:
            min_val = float(attr_min)

    if len(node.input) >= 3 and node.input[2]:
        if node.input[2] not in constants:
            return False
        max_arr = np.asarray(constants[node.input[2]], dtype=np.float64)
        if max_arr.size != 1:
            return False
        max_val = float(max_arr.reshape(()))
    else:
        attr_max = _get_attr(node, "max", None)
        if attr_max is not None:
            max_val = float(attr_max)

    return min_val == 0.0 and (max_val is None or np.isposinf(max_val))


def _collect_constants(graph: onnx.GraphProto) -> tuple[dict[str, np.ndarray], int]:
    """Collect graph initializers and constant-folded helper tensors."""
    constants: dict[str, np.ndarray] = {}
    init_param_count = 0
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        constants[init.name] = arr
        if init.data_type in (
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        ):
            init_param_count += int(arr.size)

    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            arr = _extract_constant_from_node(node)
            if arr is not None:
                constants[node.output[0]] = arr
        elif node.op_type == "Identity" and node.input and node.output:
            if node.input[0] in constants:
                constants[node.output[0]] = constants[node.input[0]]
        elif node.op_type == "Transpose" and node.input and node.output:
            if node.input[0] in constants:
                perm = _get_attr(node, "perm", None)
                constants[node.output[0]] = np.transpose(constants[node.input[0]], axes=perm)
        elif node.op_type == "Cast" and node.input and node.output and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
    return constants, init_param_count


def _parse_sequential_mlp(
    model_proto: onnx.ModelProto,
    constants: dict[str, np.ndarray],
) -> tuple[list[DenseLayer] | None, list[str]]:
    """Attempt exact sequential MLP extraction from an ONNX graph.

    Returns (layers, op_types) where layers is None if the graph is not a pure
    sequential dense MLP with activations in {"Relu", "Tanh", "Sigmoid", "Identity"}.
    """
    graph = model_proto.graph
    init_names = {init.name for init in graph.initializer}
    data_inputs = [vi.name for vi in graph.input if vi.name not in init_names]
    if len(data_inputs) != 1 or len(graph.output) != 1:
        return None, []

    cur_tensor = data_inputs[0]
    target_output = graph.output[0].name

    # Build consumer map ignoring pure constant-folding nodes.
    active_nodes: list[onnx.NodeProto] = []
    consumers: dict[str, int] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            continue
        if (
            node.op_type in ("Identity", "Transpose", "Cast")
            and node.input
            and node.input[0] in constants
        ):
            continue
        active_nodes.append(node)
        for inp in node.input:
            if inp and inp not in constants:
                consumers[inp] = consumers.get(inp, 0) + 1

    layers: list[DenseLayer] = []
    op_types: list[str] = []
    pending_activation_allowed = False
    idx = 0

    while idx < len(active_nodes):
        node = active_nodes[idx]
        op = node.op_type

        if op == "Identity" and len(node.input) == 1 and node.input[0] == cur_tensor:
            if consumers.get(cur_tensor, 0) != 1:
                return None, []
            cur_tensor = node.output[0]
            idx += 1
            continue

        if op == "Gemm":
            if len(node.input) < 2 or node.input[0] != cur_tensor:
                return None, []
            if consumers.get(cur_tensor, 0) != 1 or node.input[1] not in constants:
                return None, []
            trans_a = _get_attr(node, "transA", 0)
            trans_b = _get_attr(node, "transB", 0)
            alpha = float(_get_attr(node, "alpha", 1.0))
            beta = float(_get_attr(node, "beta", 1.0))
            if trans_a != 0:
                return None, []

            raw_w = np.asarray(constants[node.input[1]], dtype=np.float64)
            if raw_w.ndim != 2:
                return None, []
            weight = alpha * (raw_w if trans_b else raw_w.T)
            out_features = weight.shape[0]

            if len(node.input) >= 3 and node.input[2]:
                if node.input[2] not in constants:
                    return None, []
                bias = _broadcast_bias(constants[node.input[2]], out_features)
                if bias is None:
                    return None, []
                bias = beta * bias
            else:
                bias = np.zeros((out_features,), dtype=np.float64)

            layer_name = node.name or f"layer_{len(layers)}"
            layers.append(DenseLayer(name=layer_name, weight=weight, bias=bias, activation="Identity"))
            op_types.append("Gemm")
            cur_tensor = node.output[0]
            pending_activation_allowed = True
            idx += 1
            continue

        if op == "MatMul":
            if len(node.input) != 2 or node.input[0] != cur_tensor:
                return None, []
            if consumers.get(cur_tensor, 0) != 1 or node.input[1] not in constants:
                return None, []
            raw_w = np.asarray(constants[node.input[1]], dtype=np.float64)
            if raw_w.ndim != 2:
                return None, []
            weight = raw_w.T.copy()
            out_features = weight.shape[0]
            bias = np.zeros((out_features,), dtype=np.float64)
            cur_tensor = node.output[0]
            layer_op = "MatMul"
            layer_name = node.name or f"layer_{len(layers)}"

            # Fuse trailing Add(mm_out, bias_const) or Add(bias_const, mm_out).
            if idx + 1 < len(active_nodes) and active_nodes[idx + 1].op_type == "Add":
                add_node = active_nodes[idx + 1]
                if (
                    consumers.get(cur_tensor, 0) == 1
                    and len(add_node.input) == 2
                    and cur_tensor in add_node.input
                ):
                    other = (
                        add_node.input[1]
                        if add_node.input[0] == cur_tensor
                        else add_node.input[0]
                    )
                    if other in constants:
                        maybe_bias = _broadcast_bias(constants[other], out_features)
                        if maybe_bias is None:
                            return None, []
                        bias = maybe_bias
                        cur_tensor = add_node.output[0]
                        layer_op = "MatMul+Add"
                        idx += 1

            layers.append(DenseLayer(name=layer_name, weight=weight, bias=bias, activation="Identity"))
            op_types.append(layer_op)
            pending_activation_allowed = True
            idx += 1
            continue

        if op in ("Relu", "Tanh", "Sigmoid", "Clip") and pending_activation_allowed:
            if not node.input or node.input[0] != cur_tensor or consumers.get(cur_tensor, 0) != 1:
                return None, []
            if op == "Clip":
                if not _clip_is_relu(node, constants):
                    return None, []
                act = "Relu"
            else:
                act = op
            layers[-1].activation = act
            cur_tensor = node.output[0]
            pending_activation_allowed = False
            idx += 1
            continue

        return None, []

    if not layers or cur_tensor != target_output:
        return None, []
    return layers, op_types


def _infer_graph_dims(
    session: ort.InferenceSession,
    mlp_layers: list[DenseLayer] | None,
) -> tuple[int, int]:
    in_shape = session.get_inputs()[0].shape
    out_shape = session.get_outputs()[0].shape

    def _last_int(shape: list[Any]) -> int | None:
        if shape and isinstance(shape[-1], int) and shape[-1] > 0:
            return int(shape[-1])
        return None

    in_dim = _last_int(list(in_shape))
    out_dim = _last_int(list(out_shape))
    if in_dim is None and mlp_layers:
        in_dim = int(mlp_layers[0].weight.shape[1])
    if out_dim is None and mlp_layers:
        out_dim = int(mlp_layers[-1].weight.shape[0])
    if in_dim is None or out_dim is None:
        raise ValueError(
            f"Unable to infer static input/output dims from ONNX shapes {in_shape} -> {out_shape}"
        )
    return in_dim, out_dim


def _load_sibling_domain(onnx_path: Path) -> DomainSpec | None:
    meta_path = onnx_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "domain" in raw:
        return DomainSpec.model_validate(raw["domain"])
    return DomainSpec.model_validate(raw)


def _default_domain(input_dim: int, output_dim: int) -> DomainSpec:
    return DomainSpec(
        feature_names=[f"x{i}" for i in range(input_dim)],
        lower_bounds=[-1.0] * input_dim,
        upper_bounds=[1.0] * input_dim,
        output_names=[f"y{i}" for i in range(output_dim)],
        task_type=TaskType.REGRESSION,
    )


class LoadedModel:
    """Loaded ONNX neural network wrapper with runtime inference and MLP extraction."""

    def __init__(
        self,
        path: Path,
        model_proto: onnx.ModelProto,
        session: ort.InferenceSession,
        metadata: ModelMetadata,
        mlp_layers: list[DenseLayer] | None,
    ) -> None:
        self.path = path
        self.model_proto = model_proto
        self.session = session
        self.metadata = metadata
        self._mlp_layers = mlp_layers
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @property
    def domain(self) -> DomainSpec:
        return self.metadata.domain

    def extract_mlp_layers(self) -> list[DenseLayer] | None:
        """Return exact feed-forward MLP DenseLayer list, or None for non-MLP graphs."""
        if self._mlp_layers is None:
            return None
        return [
            DenseLayer(
                name=layer.name,
                weight=layer.weight.copy(),
                bias=layer.bias.copy(),
                activation=layer.activation,
            )
            for layer in self._mlp_layers
        ]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference and return float64 outputs of shape (N, output_dim)."""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        if arr.shape[1] != self.metadata.input_dim:
            raise ValueError(
                f"Expected {self.metadata.input_dim} input features, got {arr.shape[1]}"
            )
        if arr.shape[0] == 0:
            return np.empty((0, self.metadata.output_dim), dtype=np.float64)

        raw = self.session.run([self.output_name], {self.input_name: arr})[0]
        out = np.asarray(raw, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Return int64 class predictions of shape (N,)."""
        out = self.predict(X)
        if len(out) == 0:
            return np.empty((0,), dtype=np.int64)
        if out.shape[1] == 1:
            return (out[:, 0] > 0.5).astype(np.int64)
        return np.argmax(out, axis=1).astype(np.int64)

    def hidden_activations(self, X: np.ndarray) -> list[np.ndarray]:
        """Compute intermediate hidden post-activation arrays for extracted MLP layers."""
        if self._mlp_layers is None or len(self._mlp_layers) <= 1:
            return []
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        if arr.shape[1] != self.metadata.input_dim:
            raise ValueError(
                f"Expected {self.metadata.input_dim} input features, got {arr.shape[1]}"
            )

        hidden_layers = self._mlp_layers[:-1]
        if len(arr) == 0:
            return [np.empty((0, layer.weight.shape[0]), dtype=np.float64) for layer in hidden_layers]

        acts: list[np.ndarray] = []
        cur = arr
        for layer in hidden_layers:
            z = cur @ layer.weight.T + layer.bias
            cur = _apply_activation(z, layer.activation)
            acts.append(cur)
        return acts

    @classmethod
    def load_onnx(
        cls,
        path: str | Path,
        domain: DomainSpec | None = None,
        name: str | None = None,
    ) -> LoadedModel:
        """Load an ONNX file, compute metadata, and parse sequential MLP layers if exact."""
        return load_onnx(path=path, domain=domain, name=name)

    @classmethod
    def from_pytorch(
        cls,
        module: Any,
        domain: DomainSpec,
        export_path: str | Path,
        name: str | None = None,
    ) -> LoadedModel:
        """Export a PyTorch module to ONNX (opset 17, dynamo=False) and load it."""
        return from_pytorch(module=module, domain=domain, export_path=export_path, name=name)


def load_onnx(
    path: str | Path,
    domain: DomainSpec | None = None,
    name: str | None = None,
) -> LoadedModel:
    """Load an ONNX model from disk into a LoadedModel."""
    onnx_path = Path(path)
    raw_bytes = onnx_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    size_bytes = len(raw_bytes)

    model_proto = onnx.load_from_string(raw_bytes)
    session = ort.InferenceSession(raw_bytes, providers=["CPUExecutionProvider"])

    constants, init_param_count = _collect_constants(model_proto.graph)
    mlp_layers, op_types = _parse_sequential_mlp(model_proto, constants)
    in_dim, out_dim = _infer_graph_dims(session, mlp_layers)

    resolved_domain = domain if domain is not None else _load_sibling_domain(onnx_path)
    if resolved_domain is None:
        resolved_domain = _default_domain(in_dim, out_dim)

    if resolved_domain.input_dim != in_dim or resolved_domain.output_dim != out_dim:
        raise ValueError(
            f"DomainSpec dimensions ({resolved_domain.input_dim} -> {resolved_domain.output_dim}) "
            f"do not match ONNX graph ({in_dim} -> {out_dim})"
        )

    if mlp_layers is not None:
        layer_infos = [
            LayerInfo(
                name=layer.name,
                op_type=op_type,
                in_features=int(layer.weight.shape[1]),
                out_features=int(layer.weight.shape[0]),
                activation=layer.activation,
                param_count=int(layer.weight.size + layer.bias.size),
            )
            for layer, op_type in zip(mlp_layers, op_types)
        ]
        total_params = sum(info.param_count for info in layer_infos)
        activations = [layer.activation for layer in mlp_layers]
        is_piecewise_linear = all(act in {"Relu", "Identity"} for act in activations)
    else:
        layer_infos = [
            LayerInfo(
                name=node.name or f"{node.op_type}_{i}",
                op_type=node.op_type,
                in_features=0,
                out_features=0,
                activation=node.op_type
                if node.op_type in ("Relu", "Tanh", "Sigmoid", "Clip")
                else "Identity",
                param_count=0,
            )
            for i, node in enumerate(model_proto.graph.node)
            if node.op_type != "Constant"
        ]
        total_params = init_param_count
        activations = [
            node.op_type
            for node in model_proto.graph.node
            if node.op_type in ("Relu", "Tanh", "Sigmoid", "Clip")
        ]
        is_piecewise_linear = False

    metadata = ModelMetadata(
        name=name or onnx_path.stem,
        path=str(onnx_path),
        sha256=sha256,
        size_bytes=size_bytes,
        input_dim=in_dim,
        output_dim=out_dim,
        total_params=total_params,
        is_piecewise_linear=is_piecewise_linear,
        activations=activations,
        layers=layer_infos,
        domain=resolved_domain,
    )
    return LoadedModel(
        path=onnx_path,
        model_proto=model_proto,
        session=session,
        metadata=metadata,
        mlp_layers=mlp_layers,
    )


def from_pytorch(
    module: Any,
    domain: DomainSpec,
    export_path: str | Path,
    name: str | None = None,
) -> LoadedModel:
    """Export a PyTorch module to ONNX with dynamo=False, opset_version=17 and load it."""
    import torch

    out_path = Path(export_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    was_training = bool(module.training)
    module.eval()
    try:
        param = next(module.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        mid = [0.5 * (lo + hi) for lo, hi in zip(domain.lower_bounds, domain.upper_bounds)]
        dummy = torch.tensor([mid], dtype=torch.float32, device=device)
        with torch.no_grad():
            torch.onnx.export(
                module,
                dummy,
                str(out_path),
                dynamo=False,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
    finally:
        if was_training:
            module.train()

    meta_path.write_text(domain.model_dump_json(indent=2), encoding="utf-8")
    return load_onnx(out_path, domain=domain, name=name)


def _format_expr_readable(expr: Expr, precision: int = 4) -> str:
    """Format an IR Expr AST into a readable deterministic algebraic string."""
    if isinstance(expr, VarExpr):
        return expr.name
    if isinstance(expr, ConstExpr):
        return f"{expr.value:.{precision}f}"
    if isinstance(expr, UnaryExpr):
        inner = _format_expr_readable(expr.operand, precision)
        if expr.op == "neg":
            return f"-({inner})"
        return f"{expr.op}({inner})"
    if isinstance(expr, ClampExpr):
        inner = _format_expr_readable(expr.operand, precision)
        return f"clamp({inner}, {expr.low:.{precision}f}, {expr.high:.{precision}f})"
    if isinstance(expr, BinaryExpr):
        if expr.op == "mul":
            if isinstance(expr.left, ConstExpr) and isinstance(expr.right, VarExpr):
                return f"{expr.left.value:.{precision}f}*{expr.right.name}"
            if isinstance(expr.right, ConstExpr) and isinstance(expr.left, VarExpr):
                return f"{expr.right.value:.{precision}f}*{expr.left.name}"
        left_s = _format_expr_readable(expr.left, precision)
        right_s = _format_expr_readable(expr.right, precision)
        if expr.op == "add":
            if right_s.startswith("-"):
                return f"{left_s} - {right_s[1:]}"
            return f"{left_s} + {right_s}"
        if expr.op == "sub":
            return f"{left_s} - ({right_s})"
        if expr.op == "mul":
            return f"({left_s}) * ({right_s})"
        if expr.op == "div":
            return f"({left_s}) / ({right_s})"
        return f"{expr.op}({left_s}, {right_s})"
    return str(expr)


def compute_regime_neuron_profiles(
    model: LoadedModel,
    program: DecompiledProgram,
    X: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Compute per-regime hidden neuron activation statistics across dataset X."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {arr.shape}")

    hidden_acts = model.hidden_activations(arr) if len(arr) > 0 else []
    _, assigned = program.predict_with_regimes(arr) if len(arr) > 0 else (
        np.empty((0, program.domain.output_dim), dtype=np.float64),
        np.empty((0,), dtype=np.int64),
    )

    total_hidden = sum(h.shape[1] for h in hidden_acts)
    profiles: dict[int, dict[str, Any]] = {}
    for leaf in program.leaves():
        rid = int(leaf.regime_id)
        idx = np.flatnonzero(assigned == rid)
        n_samples = len(idx)

        if n_samples == 0 or not hidden_acts:
            profiles[rid] = {
                "regime_id": rid,
                "regime_name": leaf.name,
                "sample_count": n_samples,
                "mean_layer_activation_rates": [],
                "layer_activation_rates": [],
                "active_neuron_count": 0,
                "active_count": 0,
                "total_hidden_neurons": total_hidden,
                "top_neurons": [],
            }
            continue

        mean_layer_rates: list[float] = []
        layer_rates_list: list[list[float]] = []
        all_neurons: list[dict[str, Any]] = []
        active_count = 0

        for layer_idx, h_mat in enumerate(hidden_acts):
            h_sub = h_mat[idx]
            rates = np.mean(h_sub > 0.0, axis=0)
            means = np.mean(h_sub, axis=0)
            mean_layer_rates.append(float(np.mean(rates)))
            layer_rates_list.append([float(r) for r in rates])
            active_count += int(np.sum(rates > 0.5))

            for neuron_idx in range(h_sub.shape[1]):
                all_neurons.append(
                    {
                        "layer": int(layer_idx),
                        "neuron": int(neuron_idx),
                        "index": int(neuron_idx),
                        "activation_rate": float(rates[neuron_idx]),
                        "mean_value": float(means[neuron_idx]),
                    }
                )

        all_neurons.sort(
            key=lambda item: (item["activation_rate"], item["mean_value"]),
            reverse=True,
        )
        top_5 = [n for n in all_neurons if n["activation_rate"] > 0.0][:5]
        if not top_5:
            top_5 = all_neurons[:5]

        profiles[rid] = {
            "regime_id": rid,
            "regime_name": leaf.name,
            "sample_count": n_samples,
            "mean_layer_activation_rates": mean_layer_rates,
            "layer_activation_rates": layer_rates_list,
            "active_neuron_count": active_count,
            "active_count": active_count,
            "total_hidden_neurons": total_hidden,
            "top_neurons": top_5,
        }

    return profiles


def explain_decision(
    model: LoadedModel,
    program: DecompiledProgram,
    x: Sequence[float],
    leaf_proven_map: dict[int, bool] | None = None,
) -> DecisionExplanation:
    """Produce a deterministic causal trace and neuron attribution for input x."""
    x_list = [float(v) for v in x]
    if len(x_list) != model.metadata.input_dim:
        raise ValueError(
            f"Expected input vector of length {model.metadata.input_dim}, got {len(x_list)}"
        )
    x_2d = np.asarray([x_list], dtype=np.float64)

    prog_out, leaf, guard_trace = program.trace_one(x_list)
    nn_out_arr = model.predict(x_2d)[0]
    nn_out = [float(v) for v in nn_out_arr]
    prog_out_f = [float(v) for v in prog_out]
    abs_err = float(np.max(np.abs(np.asarray(nn_out) - np.asarray(prog_out_f))))

    nn_class: int | None = None
    program_class: int | None = None
    if program.task_type == TaskType.CLASSIFICATION:
        nn_class = int(model.predict_classes(x_2d)[0])
        program_class = int(program.predict_classes(x_2d)[0])

    formula_trace: list[dict[str, Any]] = []
    out_names = program.domain.output_names
    for k, expr in enumerate(leaf.output_exprs):
        out_name = out_names[k] if k < len(out_names) else f"y{k}"
        expr_str = _format_expr_readable(expr, precision=4)
        val = prog_out_f[k] if k < len(prog_out_f) else 0.0
        nn_val = nn_out[k] if k < len(nn_out) else 0.0
        formula_trace.append(
            {
                "output_index": k,
                "output_name": out_name,
                "expression": expr_str,
                "formula": f"{out_name} = {expr_str}",
                "value": val,
                "nn_value": nn_val,
            }
        )

    hidden_acts = model.hidden_activations(x_2d)
    active_neurons: list[dict[str, Any]] = []
    neuron_summary_parts: list[str] = []
    for layer_idx, h_mat in enumerate(hidden_acts):
        h_vec = h_mat[0]
        active_idx = np.flatnonzero(h_vec > 0.0)
        sorted_active = active_idx[np.argsort(h_vec[active_idx])[::-1]]
        top_neurons = [
            {
                "index": int(j),
                "neuron": int(j),
                "value": float(h_vec[j]),
            }
            for j in sorted_active[:5]
        ]
        active_neurons.append(
            {
                "layer": int(layer_idx),
                "active_count": len(active_idx),
                "total_count": len(h_vec),
                "top_neurons": top_neurons,
            }
        )
        neuron_summary_parts.append(f"L{layer_idx} {len(active_idx)}/{len(h_vec)}")

    rid = int(leaf.regime_id)
    is_proven = bool(leaf_proven_map.get(rid, False)) if leaf_proven_map else False
    badge = "PROVEN" if is_proven else "EMPIRICAL"

    if guard_trace:
        guard_steps = " -> ".join(
            f"[{g['expression']} is {g['branch']} (margin={float(g['margin']):.4f})]"
            for g in guard_trace
        )
        route_str = f"Guards {guard_steps} routed input to Regime #{rid} ({leaf.name}) [{badge}]."
    else:
        route_str = f"Single root Regime #{rid} ({leaf.name}) [{badge}]."

    if formula_trace:
        f0 = formula_trace[0]
        out_str = (
            f"Evaluated {f0['formula']} = {f0['value']:.4f} "
            f"(ONNX: {f0['nn_value']:.4f}, max |err|={abs_err:.4e})."
        )
    else:
        out_str = f"Max |err|={abs_err:.4e}."

    if program.task_type == TaskType.CLASSIFICATION and program_class is not None:
        cls_names = program.domain.class_names
        p_label = cls_names[program_class] if 0 <= program_class < len(cls_names) else str(program_class)
        out_str += f" Predicted class: {p_label} (id={program_class}, NN id={nn_class})."

    if neuron_summary_parts:
        out_str += f" Active hidden neurons: {', '.join(neuron_summary_parts)}."

    return DecisionExplanation(
        x=x_list,
        regime_id=rid,
        regime_name=leaf.name,
        regime_proven=is_proven,
        guard_trace=guard_trace,
        formula_trace=formula_trace,
        nn_output=nn_out,
        program_output=prog_out_f,
        nn_class=nn_class,
        program_class=program_class,
        abs_error=abs_err,
        active_neurons=active_neurons,
        summary_text=f"{route_str} {out_str}",
    )


__all__ = [
    "DenseLayer",
    "LoadedModel",
    "compute_regime_neuron_profiles",
    "explain_decision",
    "from_pytorch",
    "load_onnx",
]
