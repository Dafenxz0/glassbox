"""Shared Pydantic v2 data contracts for Glassbox."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from glassbox.ir import DecompileResult


class TaskType(str, Enum):
    """Supported supervised task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class VerificationStatus(str, Enum):
    """Verification outcome status for a decompiled program."""

    PROVEN_REAL_GLOBAL = "PROVEN_REAL_GLOBAL"
    PROVEN_REAL_MARGIN = "PROVEN_REAL_MARGIN"
    PROVEN_SMT = "PROVEN_SMT"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    BOUNDED_EMPIRICAL = "BOUNDED_EMPIRICAL"
    COUNTEREXAMPLE_FOUND = "COUNTEREXAMPLE_FOUND"
    UNVERIFIED = "UNVERIFIED"

    @property
    def is_proven(self) -> bool:
        """Return True if the status corresponds to a formal SMT proof."""
        return self in {
            VerificationStatus.PROVEN_REAL_GLOBAL,
            VerificationStatus.PROVEN_REAL_MARGIN,
            VerificationStatus.PROVEN_SMT,
        }


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
    measured_throughput_ips: float = 0.0
    batch_size: int = 1
    native_loop_p50_ns: float | None = None
    source_bytes: int = 0
    binary_bytes: int = 0
    compiler: str = ""


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


class RealArithmeticProof(BaseModel):
    """Formal Z3 QF_LRA proof metadata over exact rational weights."""

    exact_rationals: bool = True
    sound_bounds_only: bool = True
    scope: Literal["global", "margin_conditioned", "none"] = "none"
    tolerance_eps: float = 0.05
    classification_margin: float = 0.0
    z3_version: str = ""
    solver_result: Literal["unsat", "sat", "timeout", "unknown", "skipped"] = "skipped"
    leaves_proven: int = 0
    leaves_total: int = 0
    leaf_proven_map: dict[int, bool] = Field(default_factory=dict)
    solve_time_ms: float = 0.0
    notes: str = ""


class NumericalDeploymentAssurance(BaseModel):
    """Empirical numerical divergence checks across float32 ONNX, IR, C99, and Python."""

    onnxruntime_samples_checked: int = 0
    ir_vs_onnx_max_diff: float = 0.0
    ir_vs_onnx_agreement: float = 0.0
    c99_compiled: bool = False
    c99_compiler: str = ""
    c99_vs_ir_max_diff: float | None = None
    python_vs_ir_max_diff: float | None = None
    float32_soundness_note: str = (
        "Real-arithmetic SMT certifies the mathematical rational MLP extracted from ONNX "
        "float32 weights; numerical deployment assurance empirically checks float32 "
        "ONNXRuntime and compiled C99/Python outputs."
    )


class ParetoCandidate(BaseModel):
    """A single complexity-vs-fidelity point on the decompilation Pareto frontier."""

    candidate_id: str
    max_regimes_budget: int
    actual_regimes: int
    ast_nodes: int
    status: VerificationStatus
    agreement_rate: float
    max_abs_error: float
    mean_abs_error: float
    c99_p50_us: float | None = None
    c99_native_ns: float | None = None
    c_source_bytes: int = 0
    is_pareto_optimal: bool = False
    program_sha256: str = ""
    code_c: str = ""
    code_rust: str = ""
    code_python: str = ""


class DecisionExplanation(BaseModel):
    """Step-by-step causal trace explaining how a single input routes and evaluates."""

    x: list[float]
    regime_id: int
    regime_name: str
    regime_proven: bool = False
    guard_trace: list[dict[str, Any]] = Field(default_factory=list)
    formula_trace: list[dict[str, Any]] = Field(default_factory=list)
    nn_output: list[float] = Field(default_factory=list)
    program_output: list[float] = Field(default_factory=list)
    nn_class: int | None = None
    program_class: int | None = None
    abs_error: float = 0.0
    active_neurons: list[dict[str, Any]] = Field(default_factory=list)
    summary_text: str = ""


class ArtifactManifest(BaseModel):
    """Reproducible manifest of emitted artifacts, SHA-256 digests, and tool versions."""

    schema_version: str = "2.0.0"
    created_at: str
    model_name: str
    files: dict[str, str]
    tool_versions: dict[str, str]
    config: DecompileConfig


class BundleVerificationReport(BaseModel):
    """Independent verification and replay report for an emitted artifact bundle."""

    verdict: Literal[
        "VERIFIED_REPRODUCED",
        "HASH_MISMATCH",
        "FALSIFIED_BY_REPLAY",
        "TIMEOUT",
        "UNKNOWN",
        "INVALID_BUNDLE",
    ]
    bundle_path: str
    hashes_valid: bool
    smt_replayed: bool
    claimed_status: VerificationStatus
    replayed_status: VerificationStatus
    numerical_checks_passed: bool
    details: list[str] = Field(default_factory=list)
    replay_time_ms: float = 0.0


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
    tool_versions: dict[str, str] = Field(default_factory=dict)
    config: DecompileConfig | None = None
    real_arithmetic_proof: RealArithmeticProof | None = None
    numerical_deployment_assurance: NumericalDeploymentAssurance | None = None


def __getattr__(name: str) -> Any:
    if name == "DecompileResult":
        from glassbox.ir import DecompileResult

        return DecompileResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArtifactManifest",
    "BenchmarkEntry",
    "BundleVerificationReport",
    "CEGISIteration",
    "Counterexample",
    "DecisionExplanation",
    "DecompileConfig",
    "DecompileResult",
    "DomainSpec",
    "LayerInfo",
    "ModelMetadata",
    "NumericalDeploymentAssurance",
    "ParetoCandidate",
    "RealArithmeticProof",
    "TaskType",
    "VerificationCertificate",
    "VerificationStatus",
]
