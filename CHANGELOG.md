# Changelog

All notable changes to GlassBox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Studio links directly to each recorded model, provides separate mobile views for code, input space and results, and displays missing measurements explicitly.
- Bundle files use portable LF endings. Verification checks the complete manifest and compares Python source with the emitter without importing the bundle. Skipped and newly resolved solver replays have distinct verdicts.
- Generated Rust removes redundant outer parentheses in assignments. IR expressions using transcendental functions select the standard library. All recorded zoo programs and their Pareto candidates compile with warnings denied.
- The website and benchmark ledger report the same four saved runs as Studio.

## [0.2.0] - 2026-11-23

### Added
- **Exact Rational SMT Soundness (`src/glassbox/verifier/smt.py`)**:
  - Replaced 8-significant-figure floating-point string rounding with exact IEEE-754 rational conversion (`fractions.Fraction(float(v)) -> z3.Q(num, den)`), preserving every non-zero weight and bias without truncation.
  - Replaced unverified numeric LP bounds with sound exact `Fraction` outward-rounded Interval Bound Propagation (IBP) over each leaf polyhedron's bounding box.
  - Added explicit `VerificationStatus` distinction between `PROVEN_REAL_GLOBAL` (`scope="global"`), `PROVEN_REAL_MARGIN` (`scope="margin_conditioned"` when `classification_margin > 0`), `TIMEOUT`, `UNKNOWN`, `BOUNDED_EMPIRICAL`, and `COUNTEREXAMPLE_FOUND`.
- **Two-Tier Certificate Guarantees (`RealArithmeticProof` & `NumericalDeploymentAssurance`)**:
  - Separated formal rational-arithmetic Z3 proofs (`RealArithmeticProof`) from empirical float32 ONNXRuntime and compiled native C99/Python equivalence checks (`NumericalDeploymentAssurance`).
- **Independent Reproducible Artifact Bundles (`glassbox verify`)**:
  - Added `write_artifact_bundle` and `verify_artifact_bundle` (`src/glassbox/certificate.py`) plus `glassbox verify <bundle.gbundle>` CLI command that independently checks SHA-256 digests (`manifest.json`), reloads `model.onnx` and `program.ir.json`, replays exact rational Z3 SMT from scratch, and verifies numerical equivalence (`VERIFIED_REPRODUCED`).
- **Synthesis Engine 2.0 & Deterministic Decision Explanations**:
  - Added exact normalized Layer-1 ReLU neuron hyperplanes (`w_j · x + b_j = 0`) as first-class oblique split candidates in `src/glassbox/regimes.py`.
  - Added deterministic rule-to-neuron attribution (`compute_regime_neuron_profiles`) and zero-LLM single-input decision explanations (`explain_decision`, `POST /api/explain`) tracing guard evaluations, active symbolic formulas, proof status, and top active hidden neurons.
  - Added multi-budget Certified Program Pareto Frontier exploration (`search_pareto_frontier`, `--pareto`).
- **Honest Native C Loop & Batched Benchmarks (`src/glassbox/benchmark.py`, `src/glassbox/emitters/c_emitter.py`)**:
  - Added `find_c_compiler()` (`gcc` -> `clang` -> `ziglang cc`) and cross-platform shared library compilation (`-fPIC -shared -lm` on Linux, `.dll` on Windows).
  - Emitted C99 exports `glassbox_bench_loop` for zero-FFI native C loop timing (`ns/call`) alongside single-call `ctypes` FFI, zero-dep Python, `ONNX Runtime (batch=1)`, `ONNX Runtime (batch=64)`, and `PyTorch (batch=1)` with real measured sustained throughput (`total_calls / total_wall_s`).
- **GitHub Pages Live Web Preview & GlassBox Studio 2.0**:
  - Added `.github/workflows/pages.yml` publishing the landing page (`https://dafenxz0.github.io/glassbox/`) and interactive Studio 2.0 (`https://dafenxz0.github.io/glassbox/studio/`).
  - Upgraded Studio with Certified Pareto Frontier explorer, interactive "Explain This Decision" tracer, per-regime neuron attribution badges, and split Real-Arithmetic Proof / Numerical Deployment Assurance certificate viewer.

## [0.1.0] - 2026-11-23

### Added
- **ONNX Frontend & Model Inspector**: Extracts layer topology, activation metadata, parameter counts, SHA-256 digests, and piecewise-linear eligibility from `.onnx` neural networks with `onnx` and `onnxruntime`.
- **CEGIS Synthesis Engine**: Counterexample-Guided Inductive Synthesis loop extracting piecewise-linear activation regimes and closed-form linear policies, repairing missed regimes directly from falsifier counterexamples.
- **Two-Tier Formal & Empirical Verification**:
  - **Tier A (`PROVEN_SMT`)**: Z3 `QF_LRA` SMT encoding over ReLU MLPs within declared bounded input domains.
  - **Tier B (`BOUNDED_EMPIRICAL`)**: Adversarial falsification (Sobol sampling plus Nelder-Mead search) for smooth/non-ReLU architectures (`Tanh`, `Sigmoid`).
- **Typed Program IR & Multi-Target Codegen**: Emits zero-dependency C99 (with `-DGLASSBOX_MAIN` test harness), warning-clean `no_std` Rust (with `#[cfg(test)]` unit tests), readable Python, Mermaid flowcharts, and JSON `VerificationCertificate` records.
- **Model Zoo, CLI & Web Studio**: Bundled benchmark policies (`cartpole_policy`, `drone_altitude_pid`, `reactor_safety_guard`, `nonlinear_actuator`), Typer CLI (`glassbox`), and FastAPI Studio.
