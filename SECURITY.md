# Security Policy

## Scope and Intended Use

GlassBox is a static and formal decompilation tool for inspecting, synthesizing, verifying, and compiling bounded-domain ONNX neural networks into readable C99, Rust, and Python programs.

### Safety and Acceptable-Use Boundaries

- **No anti-cheat evasion or covert deployment**: GlassBox must not be used to bypass anti-cheat protections, game client integrity checks, platform access controls, or competitive matchmaking restrictions.
- **No unverified safety-critical substitution**: While Tier A (`PROVEN_SMT`) provides an exact Z3 SMT proof within the declared input hyperrectangle and tolerance, verification certificates apply only to the exact ONNX artifact SHA-256, input domain bounds, and tolerance recorded in the certificate. Do not deploy generated code into safety-critical control loops without domain-specific certification and independent validation.

## ONNX Protobuf and Untrusted Model Notice

GlassBox parses `.onnx` files using `onnx` and executes reference evaluations with `onnxruntime`.

- **Untrusted model files**: ONNX artifacts are serialized Protocol Buffers (`ModelProto`) that may contain malformed shapes, extreme tensor allocations, unsupported custom operators, or deeply nested subgraphs. Only decompile `.onnx` files from trusted sources.
- **Resource bounds**: Large models or high-dimensional input domains can trigger high memory use or solver timeouts during SMT checking. Run untrusted or unfamiliar models inside an isolated environment with appropriate memory and CPU limits.
- **Generated code inspection**: Review emitted C99, Rust, and Python files—and verify the accompanying `VerificationCertificate` SHA-256 and status—before integrating generated artifacts into downstream builds.

## Reporting a Vulnerability

If you discover a security issue (for example: path traversal in Studio/API endpoints, command injection in compiler invocation, unsafe deserialization, or a soundness bug where `PROVEN_SMT` fails to detect a genuine counterexample), please report it responsibly:

1. **Do not** open a public GitHub issue with exploit details before a fix is available.
2. Prepare a concise report including:
   - Affected file paths, symbols, or CLI/API endpoints
   - Minimal reproduction steps (and an `.onnx` / domain spec reproducer if applicable)
   - Observed vs. expected behavior and potential impact
3. Send the report privately to the project maintainers via GitHub Security Advisories (Private Vulnerability Reporting) on this repository.

We will acknowledge receipt, investigate reproducibility, and coordinate a fix and disclosure timeline before publishing an advisory.
