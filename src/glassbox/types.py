"""Shared Pydantic v2 data contracts for Glassbox."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from glassbox.ir import DecompileResult


class TaskType(str, Enum):
    """Supported supervised task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class VerificationStatus(str, Enum):
    """Verification outcome status for a decompiled program."""

    PROVEN_SMT = "PROVEN_SMT"
    BOUNDED_EMPIRICAL = "BOUNDED_EMPIRICAL"
    COUNTEREXAMPLE_FOUND = "COUNTEREXAMPLE_FOUND"
    UNVERIFIED = "UNVERIFIED"


class DomainSpec(BaseModel):
    """Bounded input/output domain specification for verification and synthesis."""

    feature_names: list[str]
    lower_bounds: list[float]
    upper_bounds: list[float]
    output_names: list[str]
    task_type: TaskType
    class_names: list[str] = Field(default_factory=list)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    @property
    def output_dim(self) -> int:
        return len(self.output_names)

    @model_validator(mode="after")
    def _validate_domain(self) -> DomainSpec:
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        if not self.output_names:
            raise ValueError("output_names must be non-empty")
        n = len(self.feature_names)
        if len(self.lower_bounds) != n or len(self.upper_bounds) != n:
            raise ValueError(
                "feature_names, lower_bounds, and upper_bounds must have equal lengths"
            )
        for i, (low, high) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            if not (low < high):
                raise ValueError(
                    f"lower_bounds[{i}] ({low}) must be strictly less than "
                    f"upper_bounds[{i}] ({high})"
                )
        return self


class LayerInfo(BaseModel):
    """Summary metadata for a single neural network layer."""

    name: str
    op_type: str
    in_features: int
    out_features: int
    activation: str
    param_count: int


class ModelMetadata(BaseModel):
    """Structural and cryptographic metadata for a source ONNX model."""

    name: str
    path: str
    sha256: str
    size_bytes: int
    input_dim: int
    output_dim: int
    total_params: int
    is_piecewise_linear: bool
    activations: list[str]
    layers: list[LayerInfo]
    domain: DomainSpec


class Counterexample(BaseModel):
    """A concrete input witnessing disagreement between NN and decompiled program."""

    iteration: int
    source: str
    x: list[float]
    nn_output: list[float]
    program_output: list[float]
    nn_class: int | None = None
    program_class: int | None = None
    max_abs_error: float
    regime_id: int
    repaired: bool = False


class CEGISIteration(BaseModel):
    """Metrics recorded for a single CEGIS synthesis/verification iteration."""

    iteration: int
    agreement_rate: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    regime_count: int
    ast_nodes: int
    status: VerificationStatus
    counterexamples_found: int
    worst_counterexample: Counterexample | None = None
    smt_time_ms: float = 0.0
    total_time_ms: float = 0.0


class BenchmarkEntry(BaseModel):
    """Runtime benchmark measurement for a single execution target."""

    runtime: str
    p50_us: float
    p95_us: float
    p99_us: float
    throughput_ips: float
    artifact_bytes: int
    speedup_vs_onnx: float
    max_diff_vs_ref: float
    available: bool = True
    note: str = ""


class VerificationCertificate(BaseModel):
    """Serializable verification certificate summarizing CEGIS and SMT evidence."""

    schema_version: str = "1.0.0"
    created_at: str
    model_name: str
    model_sha256: str
    program_sha256: str
    status: VerificationStatus
    tolerance_eps: float
    classification_margin: float
    domain: DomainSpec
    agreement_rate: float
    max_abs_error: float
    mean_abs_error: float
    smt_checked: bool
    smt_proven: bool
    smt_timeout_s: float
    total_counterexamples_found: int
    cegis_iterations: list[CEGISIteration]
    benchmarks: list[BenchmarkEntry] = Field(default_factory=list)


class DecompileConfig(BaseModel):
    """Configuration parameters for the Glassbox decompiler pipeline."""

    task_type: TaskType | None = None
    max_regimes: int = 8
    max_depth: int = 4
    regime_method: str = "oblique_tree"
    max_degree: int = 1
    tolerance_eps: float = 0.05
    classification_margin: float = 0.05
    target_agreement: float = 0.99
    max_cegis_iters: int = 6
    initial_samples: int = 2000
    verify_smt: bool = True
    smt_timeout_s: float = 4.0
    adversarial_restarts: int = 24
    seed: int = 42


def __getattr__(name: str) -> Any:
    if name == "DecompileResult":
        from glassbox.ir import DecompileResult

        return DecompileResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkEntry",
    "CEGISIteration",
    "Counterexample",
    "DecompileConfig",
    "DecompileResult",
    "DomainSpec",
    "LayerInfo",
    "ModelMetadata",
    "TaskType",
    "VerificationCertificate",
    "VerificationStatus",
]
