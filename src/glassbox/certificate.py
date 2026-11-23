"""Verification certificate generation and SHA-256 integrity checking."""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from glassbox.ir import DecompiledProgram
from glassbox.types import (
    BenchmarkEntry,
    CEGISIteration,
    DecompileConfig,
    ModelMetadata,
    VerificationCertificate,
    VerificationStatus,
)

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel


def compute_file_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
) -> VerificationCertificate:
    """Construct a signed-by-hash VerificationCertificate for a decompiled program."""
    metadata = model if isinstance(model, ModelMetadata) else model.metadata
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return VerificationCertificate(
        schema_version="1.0.0",
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
    return not (
        cert.status == VerificationStatus.PROVEN_SMT
        and not (cert.smt_checked and cert.smt_proven)
    )


__all__ = ["build_certificate", "compute_file_sha256", "verify_certificate_integrity"]
