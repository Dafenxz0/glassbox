# Changelog

All notable changes to GlassBox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-23

### Added
- **ONNX Frontend & Model Inspector**: Extracts layer topology, activation metadata, parameter counts, SHA-256 digests, and piecewise-linear eligibility from `.onnx` neural networks with `onnx` and `onnxruntime`.
- **CEGIS Synthesis Engine**: Counterexample-Guided Inductive Synthesis loop extracting piecewise-linear activation regimes and closed-form linear policies, repairing missed regimes directly from falsifier counterexamples.
- **Two-Tier Formal & Empirical Verification**:
  - **Tier A (`PROVEN_SMT`)**: Exact Z3 `QF_LRA` SMT encoding over ReLU MLPs within declared bounded input domains, producing concrete counterexamples or a mathematical proof of equivalence within `epsilon`.
  - **Tier B (`BOUNDED_EMPIRICAL`)**: Adversarial falsification (Sobol/LHS sampling plus projected gradient ascent) for smooth/non-ReLU architectures (`Tanh`, `Sigmoid`, `GELU`, `SiLU`, etc.) where exact `QF_LRA` encoding does not apply.
- **Typed Program IR & Multi-Target Codegen**: Emits zero-dependency C99 (with `-DGLASSBOX_MAIN` test harness), warning-clean Rust (with `#[cfg(test)]` unit tests), readable Python, and JSON `VerificationCertificate` records.
- **Model Zoo & Benchmark Suite**: Bundled benchmark policies (`cartpole_policy`, `drone_controller`, `credit_scorer`, `actuator_smooth`, etc.) and reproducible latency/fidelity/artifact-size measurement scripts.
- **CLI (`glassbox`) & Web Studio (`FastAPI`)**: Command-line interface (`inspect`, `decompile`, `verify`, `bench`, `serve`) and interactive browser Studio with live CEGIS trace inspection and downloadable generated code.
- **GitHub Actions CI**: Matrix testing on Python 3.11 and 3.12 with `ruff`, `pytest`, and native `gcc` C99 + `rustc` compilation smoke verification.
