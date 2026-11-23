"""Verification certificates, SHA-256 integrity checks, and reproducible artifact bundles."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import onnxruntime as ort
import scipy
import z3

from glassbox.ir import DecompiledProgram, DecompileResult
from glassbox.types import (
    ArtifactManifest,
    BenchmarkEntry,
    BundleVerificationReport,
    CEGISIteration,
    DecompileConfig,
    ModelMetadata,
    NumericalDeploymentAssurance,
    RealArithmeticProof,
    TaskType,
    VerificationCertificate,
    VerificationStatus,
)
from glassbox.verifier.smt import verify_smt

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel


def compute_file_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_tool_versions() -> dict[str, str]:
    """Collect runtime tool and library versions for reproducibility manifests."""
    return {
        "glassbox": "0.2.0",
        "python": platform.python_version(),
        "z3": z3.get_version_string(),
        "onnxruntime": getattr(ort, "__version__", "unknown"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def build_numerical_assurance(
    ir_vs_onnx_max_diff: float,
    ir_vs_onnx_agreement: float,
    onnxruntime_samples_checked: int = 4096,
    benchmarks: list[BenchmarkEntry] | None = None,
) -> NumericalDeploymentAssurance:
    """Construct Tier-2 NumericalDeploymentAssurance from empirical & benchmark runs."""
    c99_compiled = False
    c99_compiler = ""
    c99_diff: float | None = None
    py_diff: float | None = None

    for b in benchmarks or []:
        if not b.available:
            continue
        rt_lower = b.runtime.lower()
        if "c99" in rt_lower:
            c99_compiled = True
            if b.compiler:
                c99_compiler = b.compiler
            c99_diff = (
                float(b.max_diff_vs_ref)
                if c99_diff is None
                else max(c99_diff, float(b.max_diff_vs_ref))
            )
        elif "python" in rt_lower:
            py_diff = float(b.max_diff_vs_ref)

    return NumericalDeploymentAssurance(
        onnxruntime_samples_checked=int(onnxruntime_samples_checked),
        ir_vs_onnx_max_diff=float(ir_vs_onnx_max_diff),
        ir_vs_onnx_agreement=float(ir_vs_onnx_agreement),
        c99_compiled=c99_compiled,
        c99_compiler=c99_compiler,
        c99_vs_ir_max_diff=c99_diff,
        python_vs_ir_max_diff=py_diff,
    )


def build_certificate(
    model: LoadedModel | ModelMetadata,
    program: DecompiledProgram,
    config: DecompileConfig,
    status: VerificationStatus,
    agreement_rate: float,
    max_abs_error: float,
    mean_abs_error: float,
    smt_checked: bool,
    smt_proven: bool,
    total_counterexamples_found: int,
    cegis_iterations: list[CEGISIteration],
    benchmarks: list[BenchmarkEntry] | None = None,
    real_arithmetic_proof: RealArithmeticProof | None = None,
    numerical_deployment_assurance: NumericalDeploymentAssurance | None = None,
    tool_versions: dict[str, str] | None = None,
    eval_samples: int = 4096,
) -> VerificationCertificate:
    """Construct a signed-by-hash VerificationCertificate for a decompiled program."""
    metadata = model if isinstance(model, ModelMetadata) else model.metadata
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    leaves = program.leaves()

    if real_arithmetic_proof is None:
        task_type = config.task_type or metadata.domain.task_type
        scope: Literal["global", "margin_conditioned", "none"] = (
            "none"
            if not smt_checked
            else (
                "margin_conditioned"
                if (
                    task_type == TaskType.CLASSIFICATION
                    and float(config.classification_margin) > 0.0
                )
                else "global"
            )
        )
        solver_res: Literal["unsat", "sat", "timeout", "unknown", "skipped"] = (
            "skipped"
            if not smt_checked
            else (
                "unsat"
                if smt_proven
                else (
                    "timeout"
                    if status == VerificationStatus.TIMEOUT
                    else ("unknown" if status == VerificationStatus.UNKNOWN else "sat")
                )
            )
        )
        real_arithmetic_proof = RealArithmeticProof(
            exact_rationals=True,
            sound_bounds_only=True,
            scope=scope,
            tolerance_eps=float(config.tolerance_eps),
            classification_margin=float(config.classification_margin),
            z3_version=z3.get_version_string(),
            solver_result=solver_res,
            leaves_proven=len(leaves) if smt_proven else 0,
            leaves_total=len(leaves),
            leaf_proven_map={lf.regime_id: bool(smt_proven) for lf in leaves},
        )

    if numerical_deployment_assurance is None:
        numerical_deployment_assurance = build_numerical_assurance(
            ir_vs_onnx_max_diff=max_abs_error,
            ir_vs_onnx_agreement=agreement_rate,
            onnxruntime_samples_checked=eval_samples,
            benchmarks=benchmarks,
        )

    return VerificationCertificate(
        schema_version="2.0.0",
        created_at=created_at,
        model_name=metadata.name,
        model_sha256=metadata.sha256,
        program_sha256=program.sha256(),
        status=status,
        tolerance_eps=float(config.tolerance_eps),
        classification_margin=float(config.classification_margin),
        domain=metadata.domain,
        agreement_rate=float(agreement_rate),
        max_abs_error=float(max_abs_error),
        mean_abs_error=float(mean_abs_error),
        smt_checked=bool(smt_checked),
        smt_proven=bool(smt_proven),
        smt_timeout_s=float(config.smt_timeout_s),
        total_counterexamples_found=int(total_counterexamples_found),
        cegis_iterations=list(cegis_iterations),
        benchmarks=list(benchmarks) if benchmarks is not None else [],
        tool_versions=dict(tool_versions) if tool_versions is not None else collect_tool_versions(),
        config=config,
        real_arithmetic_proof=real_arithmetic_proof,
        numerical_deployment_assurance=numerical_deployment_assurance,
    )


def verify_certificate_integrity(
    cert: VerificationCertificate,
    model_path: str | Path,
    program: DecompiledProgram,
) -> bool:
    """Verify that model SHA-256, program SHA-256, and SMT flags are intact and consistent."""
    path = Path(model_path)
    if not path.is_file():
        return False
    actual_model_sha = compute_file_sha256(path)
    if not hmac.compare_digest(actual_model_sha, cert.model_sha256):
        return False
    actual_program_sha = program.sha256()
    if not hmac.compare_digest(actual_program_sha, cert.program_sha256):
        return False
    if cert.smt_proven and not cert.smt_checked:
        return False
    return not (cert.status.is_proven and not (cert.smt_checked and cert.smt_proven))


def write_artifact_bundle(
    result: DecompileResult,
    model_path: str | Path,
    bundle_dir: str | Path,
) -> Path:
    """Write a self-contained reproducible artifact bundle for `glassbox verify`."""
    from glassbox.emitters import emit_all

    bdir = Path(bundle_dir)
    code_dir = bdir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    src_onnx = Path(model_path)
    dst_onnx = bdir / "model.onnx"
    if src_onnx.resolve() != dst_onnx.resolve():
        shutil.copy2(src_onnx, dst_onnx)

    code_map = dict(result.code) if result.code else {}
    if not {"c", "h", "rust", "python"}.issubset(code_map.keys()):
        code_map = {**emit_all(result.program), **code_map}

    (bdir / "program.ir.json").write_text(
        result.program.model_dump_json(indent=2), encoding="utf-8"
    )
    (bdir / "certificate.json").write_text(
        result.certificate.model_dump_json(indent=2), encoding="utf-8"
    )
    (code_dir / "model.c").write_text(code_map["c"], encoding="utf-8")
    (code_dir / "model.h").write_text(code_map["h"], encoding="utf-8")
    (code_dir / "model.rs").write_text(code_map["rust"], encoding="utf-8")
    (code_dir / "model.py").write_text(code_map["python"], encoding="utf-8")

    rel_files = [
        "model.onnx",
        "program.ir.json",
        "certificate.json",
        "code/model.c",
        "code/model.h",
        "code/model.rs",
        "code/model.py",
    ]
    files_sha: dict[str, str] = {
        rel: compute_file_sha256(bdir / rel) for rel in rel_files
    }

    manifest = ArtifactManifest(
        schema_version="2.0.0",
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        model_name=result.metadata.name,
        files=files_sha,
        tool_versions=result.certificate.tool_versions or collect_tool_versions(),
        config=result.certificate.config
        or DecompileConfig(
            tolerance_eps=result.certificate.tolerance_eps,
            classification_margin=result.certificate.classification_margin,
            smt_timeout_s=result.certificate.smt_timeout_s,
        ),
    )
    (bdir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return bdir


def verify_artifact_bundle(
    bundle_dir: str | Path,
    recompute_smt: bool = True,
    smt_timeout_s: float | None = None,
) -> BundleVerificationReport:
    """Independently audit SHA-256 digests, replay exact Z3 SMT, and check numerical outputs."""
    from glassbox.loader import load_onnx

    t0 = time.perf_counter()
    bdir = Path(bundle_dir)
    required = ["manifest.json", "certificate.json", "model.onnx", "program.ir.json"]
    missing = [rel for rel in required if not (bdir / rel).is_file()]
    if missing:
        return BundleVerificationReport(
            verdict="INVALID_BUNDLE",
            bundle_path=str(bdir),
            hashes_valid=False,
            smt_replayed=False,
            claimed_status=VerificationStatus.UNVERIFIED,
            replayed_status=VerificationStatus.UNVERIFIED,
            numerical_checks_passed=False,
            details=[f"Missing required bundle files: {', '.join(missing)}"],
            replay_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    try:
        manifest = ArtifactManifest.model_validate_json(
            (bdir / "manifest.json").read_text(encoding="utf-8")
        )
        cert = VerificationCertificate.model_validate_json(
            (bdir / "certificate.json").read_text(encoding="utf-8")
        )
        program = DecompiledProgram.model_validate_json(
            (bdir / "program.ir.json").read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001
        return BundleVerificationReport(
            verdict="INVALID_BUNDLE",
            bundle_path=str(bdir),
            hashes_valid=False,
            smt_replayed=False,
            claimed_status=VerificationStatus.UNVERIFIED,
            replayed_status=VerificationStatus.UNVERIFIED,
            numerical_checks_passed=False,
            details=[f"Malformed bundle JSON: {exc}"],
            replay_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    details: list[str] = []
    hashes_valid = True
    for rel, expected_sha in manifest.files.items():
        fpath = bdir / rel
        if not fpath.is_file():
            hashes_valid = False
            details.append(f"Missing manifest file: {rel}")
            continue
        actual_sha = compute_file_sha256(fpath)
        if not hmac.compare_digest(actual_sha, expected_sha):
            hashes_valid = False
            details.append(f"SHA-256 mismatch for {rel}: expected {expected_sha[:12]}..., got {actual_sha[:12]}...")

    if not verify_certificate_integrity(cert, bdir / "model.onnx", program):
        hashes_valid = False
        details.append("Certificate model_sha256 or program_sha256 does not match bundle contents.")

    if not hashes_valid:
        return BundleVerificationReport(
            verdict="HASH_MISMATCH",
            bundle_path=str(bdir),
            hashes_valid=False,
            smt_replayed=False,
            claimed_status=cert.status,
            replayed_status=VerificationStatus.UNVERIFIED,
            numerical_checks_passed=False,
            details=details,
            replay_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    details.append(f"Verified SHA-256 digests for all {len(manifest.files)} bundle files.")

    try:
        model = load_onnx(bdir / "model.onnx", domain=cert.domain, name=cert.model_name)
    except Exception as exc:  # noqa: BLE001
        return BundleVerificationReport(
            verdict="INVALID_BUNDLE",
            bundle_path=str(bdir),
            hashes_valid=True,
            smt_replayed=False,
            claimed_status=cert.status,
            replayed_status=VerificationStatus.UNVERIFIED,
            numerical_checks_passed=False,
            details=[*details, f"Failed to load model.onnx: {exc}"],
            replay_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # Empirical numerical check: ONNXRuntime vs IR + emitted Python vs IR.
    rng = np.random.default_rng(42)
    lo = np.asarray(cert.domain.lower_bounds, dtype=np.float64)
    hi = np.asarray(cert.domain.upper_bounds, dtype=np.float64)
    x_check = rng.uniform(lo, hi, size=(128, cert.domain.input_dim))
    nn_y = np.asarray(model.predict(x_check), dtype=np.float64)
    ir_y = program.predict(x_check)
    max_diff = float(np.max(np.abs(nn_y - ir_y)))

    numerical_ok = True
    if cert.domain.task_type == TaskType.REGRESSION and cert.status.is_proven:
        if max_diff > cert.tolerance_eps + 1e-5:
            numerical_ok = False
            details.append(
                f"Empirical ORT vs IR diff {max_diff:.6f} exceeds proven tolerance_eps {cert.tolerance_eps:.6f}."
            )
    elif cert.domain.task_type == TaskType.CLASSIFICATION and cert.status.is_proven:
        nn_cls = model.predict_classes(x_check)
        ir_cls = program.predict_classes(x_check)
        if cert.status == VerificationStatus.PROVEN_REAL_GLOBAL and not np.array_equal(nn_cls, ir_cls):
            numerical_ok = False
            details.append("Empirical class disagreement found against PROVEN_REAL_GLOBAL claim.")

    py_path = bdir / "code" / "model.py"
    if py_path.is_file():
        spec = importlib.util.spec_from_file_location("glassbox_bundle_model", py_path)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.modules.pop(spec.name, None)
            infer_fn = getattr(mod, "infer", None)
            if callable(infer_fn):
                py_outs = np.asarray(
                    [infer_fn(row.tolist()) for row in x_check[:32]], dtype=np.float64
                )
                py_ir_diff = float(np.max(np.abs(py_outs - ir_y[:32])))
                if py_ir_diff > 1e-6:
                    numerical_ok = False
                    details.append(f"Emitted Python vs IR divergence {py_ir_diff:.3e} > 1e-6.")
                else:
                    details.append(f"Emitted Python vs IR max diff: {py_ir_diff:.2e}.")

    cfg = (cert.config or manifest.config).model_copy(
        update={
            "tolerance_eps": cert.tolerance_eps,
            "classification_margin": cert.classification_margin,
            "verify_smt": True,
            "smt_timeout_s": float(smt_timeout_s)
            if smt_timeout_s is not None
            else float(cert.smt_timeout_s),
        }
    )

    smt_replayed = False
    replayed_status = cert.status
    if recompute_smt and (cert.smt_checked or cert.status.is_proven):
        smt_res = verify_smt(model, program, cfg, iteration=0)
        smt_replayed = True
        replayed_status = smt_res.status
        details.append(
            f"Z3 SMT replay result: status={replayed_status.value}, time={smt_res.time_ms:.1f}ms."
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not numerical_ok:
        return BundleVerificationReport(
            verdict="FALSIFIED_BY_REPLAY",
            bundle_path=str(bdir),
            hashes_valid=True,
            smt_replayed=smt_replayed,
            claimed_status=cert.status,
            replayed_status=replayed_status,
            numerical_checks_passed=False,
            details=details,
            replay_time_ms=elapsed_ms,
        )

    if smt_replayed and cert.status.is_proven:
        if replayed_status == VerificationStatus.COUNTEREXAMPLE_FOUND:
            verdict: Literal[
                "VERIFIED_REPRODUCED",
                "HASH_MISMATCH",
                "FALSIFIED_BY_REPLAY",
                "TIMEOUT",
                "UNKNOWN",
                "INVALID_BUNDLE",
            ] = "FALSIFIED_BY_REPLAY"
        elif replayed_status == VerificationStatus.TIMEOUT:
            verdict = "TIMEOUT"
        elif replayed_status == VerificationStatus.UNKNOWN:
            verdict = "UNKNOWN"
        elif not replayed_status.is_proven or (
            cert.status == VerificationStatus.PROVEN_REAL_GLOBAL
            and replayed_status != VerificationStatus.PROVEN_REAL_GLOBAL
        ):
            verdict = "FALSIFIED_BY_REPLAY"
        else:
            verdict = "VERIFIED_REPRODUCED"
    else:
        verdict = "VERIFIED_REPRODUCED"

    return BundleVerificationReport(
        verdict=verdict,
        bundle_path=str(bdir),
        hashes_valid=True,
        smt_replayed=smt_replayed,
        claimed_status=cert.status,
        replayed_status=replayed_status,
        numerical_checks_passed=numerical_ok,
        details=details,
        replay_time_ms=elapsed_ms,
    )


__all__ = [
    "build_certificate",
    "build_numerical_assurance",
    "collect_tool_versions",
    "compute_file_sha256",
    "verify_artifact_bundle",
    "verify_certificate_integrity",
    "write_artifact_bundle",
]
