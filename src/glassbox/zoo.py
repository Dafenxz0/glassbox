"""Deterministic benchmark model zoo training and ONNX export."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

from glassbox.types import DomainSpec, TaskType


@dataclass(frozen=True)
class _ZooTrainingSpec:
    model_id: str
    title: str
    description: str
    architecture: list[int]
    activations: list[str]
    is_piecewise_linear: bool
    seed: int
    epochs: int
    lr: float
    domain: DomainSpec
    recommended_config: dict[str, Any]


ZOO_SPECS: dict[str, dict[str, Any]] = {}


def _build_specs() -> dict[str, _ZooTrainingSpec]:
    specs = [
        _ZooTrainingSpec(
            model_id="cartpole_policy",
            title="CartPole LQR Policy",
            description=(
                "2-class optimal LQR CartPole feedback controller (4 -> 12 -> 12 -> 2 ReLU MLP) "
                "trained on score = -0.5*x - 0.8*v + 3.2*theta + 0.9*omega."
            ),
            architecture=[4, 12, 12, 2],
            activations=["Relu", "Relu", "Identity"],
            is_piecewise_linear=True,
            seed=101,
            epochs=350,
            lr=1.5e-2,
            domain=DomainSpec(
                feature_names=["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"],
                lower_bounds=[-1.5, -2.5, -0.24, -2.5],
                upper_bounds=[1.5, 2.5, 0.24, 2.5],
                output_names=["logit_left", "logit_right"],
                task_type=TaskType.CLASSIFICATION,
                class_names=["LEFT", "RIGHT"],
            ),
            recommended_config={
                "task_type": "classification",
                "max_regimes": 6,
                "max_depth": 3,
                "max_degree": 1,
                "tolerance_eps": 0.05,
                "classification_margin": 0.05,
                "verify_smt": True,
                "smt_timeout_s": 4.0,
            },
        ),
        _ZooTrainingSpec(
            model_id="drone_altitude_pid",
            title="Drone Altitude Hold PD + Battery Sag",
            description=(
                "1D continuous thrust controller (3 -> 12 -> 10 -> 1 ReLU MLP) trained on "
                "piecewise-linear altitude-hold PD with battery-sag compensation clamped to [0, 1]."
            ),
            architecture=[3, 12, 10, 1],
            activations=["Relu", "Relu", "Identity"],
            is_piecewise_linear=True,
            seed=202,
            epochs=450,
            lr=1.2e-2,
            domain=DomainSpec(
                feature_names=["alt_error", "vert_vel", "battery_v"],
                lower_bounds=[-3.0, -3.0, 10.5],
                upper_bounds=[3.0, 3.0, 12.6],
                output_names=["thrust_cmd"],
                task_type=TaskType.REGRESSION,
            ),
            recommended_config={
                "task_type": "regression",
                "max_regimes": 8,
                "max_depth": 4,
                "max_degree": 1,
                "tolerance_eps": 0.03,
                "verify_smt": True,
                "smt_timeout_s": 4.0,
            },
        ),
        _ZooTrainingSpec(
            model_id="reactor_safety_guard",
            title="Reactor Thermal/Pressure Safety Guard",
            description=(
                "3-class reactor safety envelope classifier (4 -> 14 -> 12 -> 3 ReLU MLP) with "
                "input standardization fused into the first linear layer for clean Z3 SMT proofs."
            ),
            architecture=[4, 14, 12, 3],
            activations=["Relu", "Relu", "Identity"],
            is_piecewise_linear=True,
            seed=303,
            epochs=450,
            lr=1.2e-2,
            domain=DomainSpec(
                feature_names=["core_temp_c", "coolant_flow", "pressure_bar", "flux_rate"],
                lower_bounds=[280.0, 20.0, 80.0, 0.0],
                upper_bounds=[420.0, 100.0, 180.0, 1.5],
                output_names=["logit_normal", "logit_throttle", "logit_scram"],
                task_type=TaskType.CLASSIFICATION,
                class_names=["NORMAL", "THROTTLE", "SCRAM"],
            ),
            recommended_config={
                "task_type": "classification",
                "max_regimes": 8,
                "max_depth": 4,
                "max_degree": 1,
                "tolerance_eps": 0.05,
                "classification_margin": 0.1,
                "verify_smt": True,
                "smt_timeout_s": 4.0,
            },
        ),
        _ZooTrainingSpec(
            model_id="nonlinear_actuator",
            title="Nonlinear Actuator Torque & Damping",
            description=(
                "2D smooth nonlinear controller (3 -> 24 -> 24 -> 2 Tanh + ReLU MLP) predicting "
                "motor_torque = sin(angle_err) - 0.35*omega - 0.25*load_torque and "
                "damping_gain = 0.4 + 0.2*angle_err^2 + 0.1*tanh(omega)."
            ),
            architecture=[3, 24, 24, 2],
            activations=["Tanh", "Relu", "Identity"],
            is_piecewise_linear=False,
            seed=404,
            epochs=550,
            lr=1.0e-2,
            domain=DomainSpec(
                feature_names=["angle_err", "omega", "load_torque"],
                lower_bounds=[-1.5, -3.0, -2.0],
                upper_bounds=[1.5, 3.0, 2.0],
                output_names=["motor_torque", "damping_gain"],
                task_type=TaskType.REGRESSION,
            ),
            recommended_config={
                "task_type": "regression",
                "max_regimes": 8,
                "max_depth": 4,
                "max_degree": 2,
                "tolerance_eps": 0.05,
                "verify_smt": False,
                "adversarial_restarts": 24,
            },
        ),
    ]
    return {s.model_id: s for s in specs}


_INTERNAL_SPECS = _build_specs()
for _mid, _spec in _INTERNAL_SPECS.items():
    ZOO_SPECS[_mid] = {
        "id": _spec.model_id,
        "name": _spec.title,
        "description": _spec.description,
        "architecture": _spec.architecture,
        "activations": _spec.activations,
        "is_piecewise_linear": _spec.is_piecewise_linear,
        "domain": _spec.domain.model_dump(mode="json"),
        "recommended_config": _spec.recommended_config,
    }


def _make_mlp(architecture: list[int], activations: list[str]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, act in enumerate(activations):
        layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        if act == "Relu":
            layers.append(nn.ReLU())
        elif act == "Tanh":
            layers.append(nn.Tanh())
        elif act != "Identity":
            raise ValueError(f"Unsupported activation: {act}")
    return nn.Sequential(*layers)


def _sample_inputs(domain: DomainSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    low = np.asarray(domain.lower_bounds, dtype=np.float64)
    high = np.asarray(domain.upper_bounds, dtype=np.float64)
    return rng.uniform(low, high, size=(n, domain.input_dim))


def _target_cartpole(X: np.ndarray) -> np.ndarray:
    score = -0.5 * X[:, 0] - 0.8 * X[:, 1] + 3.2 * X[:, 2] + 0.9 * X[:, 3]
    return (score >= 0.0).astype(np.int64)


def _target_drone(X: np.ndarray) -> np.ndarray:
    alt_err, vert_vel, batt = X[:, 0], X[:, 1], X[:, 2]
    pd = 0.52 + 0.18 * alt_err - 0.14 * vert_vel
    sag = 0.06 * np.maximum(0.0, 11.8 - batt) - 0.02 * np.maximum(0.0, batt - 11.8)
    return np.clip(pd + sag, 0.0, 1.0)[:, None]


def _target_reactor(X: np.ndarray) -> np.ndarray:
    temp, flow, press, flux = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    scram = (
        (temp > 385.0)
        | (press > 160.0)
        | ((temp > 365.0) & (flow < 40.0))
        | ((flux > 1.25) & (temp > 350.0))
    )
    throttle = (~scram) & (
        (temp > 345.0) | (press > 135.0) | (flow < 50.0) | (flux > 0.95)
    )
    labels = np.zeros(len(X), dtype=np.int64)
    labels[throttle] = 1
    labels[scram] = 2
    return labels


def _target_nonlinear(X: np.ndarray) -> np.ndarray:
    angle_err, omega, load = X[:, 0], X[:, 1], X[:, 2]
    torque = np.sin(angle_err) - 0.35 * omega - 0.25 * load
    damping = 0.4 + 0.2 * (angle_err**2) + 0.1 * np.tanh(omega)
    return np.stack([torque, damping], axis=1)


_TARGET_FNS = {
    "cartpole_policy": _target_cartpole,
    "drone_altitude_pid": _target_drone,
    "reactor_safety_guard": _target_reactor,
    "nonlinear_actuator": _target_nonlinear,
}


def _fuse_input_normalization(
    net: nn.Sequential, mu: np.ndarray, sigma: np.ndarray
) -> None:
    """Fold x_norm = (x - mu) / sigma directly into the first Linear layer."""
    first_linear = net[0]
    if not isinstance(first_linear, nn.Linear):
        raise TypeError("Expected first layer to be nn.Linear")
    with torch.no_grad():
        mu_t = torch.tensor(mu, dtype=first_linear.weight.dtype)
        sigma_t = torch.tensor(sigma, dtype=first_linear.weight.dtype)
        w_scaled = first_linear.weight / sigma_t.unsqueeze(0)
        b_shifted = first_linear.bias - (w_scaled @ mu_t)
        first_linear.weight.copy_(w_scaled)
        first_linear.bias.copy_(b_shifted)


def _train_single_model(spec: _ZooTrainingSpec, models_dir: Path) -> tuple[Path, Path]:
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed)

    domain = spec.domain
    low = np.asarray(domain.lower_bounds, dtype=np.float64)
    high = np.asarray(domain.upper_bounds, dtype=np.float64)
    mu = 0.5 * (low + high)
    sigma = 0.5 * (high - low)

    X_train = _sample_inputs(domain, 6000, rng)
    X_val = _sample_inputs(domain, 1000, rng)
    target_fn = _TARGET_FNS[spec.model_id]
    Y_train = target_fn(X_train)
    Y_val = target_fn(X_val)

    X_train_norm = torch.tensor((X_train - mu) / sigma, dtype=torch.float32)

    net = _make_mlp(spec.architecture, spec.activations)
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.epochs)

    is_cls = domain.task_type == TaskType.CLASSIFICATION
    if is_cls:
        loss_fn: nn.Module = nn.CrossEntropyLoss()
        Y_train_t = torch.tensor(Y_train, dtype=torch.long)
    else:
        loss_fn = nn.MSELoss()
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32)

    net.train()
    for _ in range(spec.epochs):
        optimizer.zero_grad()
        pred = net(X_train_norm)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Fold standardization into layer 0 before ONNX export so graph is pure Gemm + Act.
    _fuse_input_normalization(net, mu, sigma)
    net.eval()

    X_val_raw = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        val_out = net(X_val_raw).numpy()

    metrics: dict[str, float] = {}
    if is_cls:
        preds = np.argmax(val_out, axis=1)
        metrics["val_accuracy"] = float(np.mean(preds == Y_val))
    else:
        diff = val_out - Y_val
        metrics["val_mse"] = float(np.mean(diff**2))
        metrics["val_mae"] = float(np.mean(np.abs(diff)))
        metrics["val_max_abs_err"] = float(np.max(np.abs(diff)))

    onnx_path = models_dir / f"{spec.model_id}.onnx"
    meta_path = models_dir / f"{spec.model_id}.meta.json"
    dummy_in = torch.tensor(mu[None, :], dtype=torch.float32)

    torch.onnx.export(
        net,
        dummy_in,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    op_types = [node.op_type for node in onnx_model.graph.node]

    raw_bytes = onnx_path.read_bytes()
    meta_payload = {
        "id": spec.model_id,
        "name": spec.title,
        "description": spec.description,
        "onnx_file": onnx_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
        "architecture": spec.architecture,
        "activations": spec.activations,
        "onnx_ops": op_types,
        "is_piecewise_linear": spec.is_piecewise_linear,
        "seed": spec.seed,
        "metrics": metrics,
        "domain": domain.model_dump(mode="json"),
        "recommended_config": spec.recommended_config,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return onnx_path, meta_path


def train_zoo(models_dir: Path, force: bool = False) -> list[Path]:
    """Train and export all benchmark zoo models to models_dir."""
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for model_id, spec in _INTERNAL_SPECS.items():
        onnx_path = target_dir / f"{model_id}.onnx"
        meta_path = target_dir / f"{model_id}.meta.json"
        if not force and onnx_path.exists() and meta_path.exists():
            exported.append(onnx_path)
            continue
        created_onnx, _ = _train_single_model(spec, target_dir)
        exported.append(created_onnx)
    return exported


def load_zoo_model_meta(models_dir: Path, model_id: str) -> dict[str, Any]:
    """Load the saved .meta.json dictionary for a zoo model."""
    target_dir = Path(models_dir)
    meta_path = target_dir / f"{model_id}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def list_zoo_models(models_dir: Path) -> list[dict[str, Any]]:
    """Return metadata records for all zoo models in canonical order."""
    target_dir = Path(models_dir)
    records: list[dict[str, Any]] = []
    for model_id, base_spec in ZOO_SPECS.items():
        onnx_path = target_dir / f"{model_id}.onnx"
        meta_path = target_dir / f"{model_id}.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["available"] = onnx_path.exists()
            meta["onnx_path"] = str(onnx_path)
            meta["meta_path"] = str(meta_path)
            records.append(meta)
        else:
            entry = dict(base_spec)
            entry["available"] = False
            entry["onnx_path"] = str(onnx_path)
            entry["meta_path"] = str(meta_path)
            records.append(entry)
    return records


__all__ = [
    "ZOO_SPECS",
    "list_zoo_models",
    "load_zoo_model_meta",
    "train_zoo",
]


if __name__ == "__main__":
    default_models_dir = Path(__file__).resolve().parents[2] / "models"
    paths = train_zoo(default_models_dir, force=True)
    for item in list_zoo_models(default_models_dir):
        print(
            f"{item['id']}: ops={item.get('onnx_ops')} "
            f"bytes={item.get('size_bytes')} metrics={item.get('metrics')}"
        )
