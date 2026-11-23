# Contributing to GlassBox

Thank you for your interest in contributing to GlassBox. This document outlines how to set up a development environment, run tests and linters, and submit changes.

## Development Setup

GlassBox requires Python 3.11 or newer. For native compiler smoke tests, `gcc` (or `clang`) and `rustc` are recommended.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

## Code Quality and Checks

Before opening a pull request, run the linter and full test suite locally:

```bash
ruff check src tests scripts
pytest -v
```

If your changes affect code generation (`c`, `rust`, or `python` targets), also verify native compilation on a real ONNX artifact:

```bash
glassbox decompile models/cartpole_policy.onnx --out /tmp/gb_out --targets c,rust,python
gcc -Wall -Wextra -Werror -std=c99 -O3 -DGLASSBOX_MAIN /tmp/gb_out/*.c -lm -o /tmp/gb_test && /tmp/gb_test
rustc --test -D warnings /tmp/gb_out/*.rs -o /tmp/gb_rs_test && /tmp/gb_rs_test
```

## Contribution Guidelines

1. **Keep changes focused**: One bug fix or feature per pull request, with minimal diff surface.
2. **Honesty in verification claims**: Never classify a heuristic or bounded empirical check as `PROVEN_SMT`. Tier A (`PROVEN_SMT`) is strictly reserved for exact Z3 SMT discharge over piecewise-linear networks within the declared domain bounds.
3. **Preserve zero-dependency codegen**: Emitted C99 (`-std=c99 -Wall -Wextra -Werror`) and Rust (`rustc -D warnings`) must compile cleanly with only standard math libraries (`-lm` for C) and zero third-party crates or headers.
4. **Update tests and docs**: Add regression tests in `tests/` for any new IR node, SMT encoding rule, or codegen edge case, and update `CHANGELOG.md` and `docs/` when user-facing behavior changes.
5. **Reproducible benchmarks**: Never hand-edit benchmark tables with estimated numbers. Regenerate benchmark reports from actual tool runs.

## Pull Request Checklist

- [ ] `ruff check src tests scripts` passes with zero errors
- [ ] `pytest -v` passes with zero failures
- [ ] Generated C99 and Rust artifacts compile cleanly with warnings treated as errors
- [ ] Documentation and docstrings match implemented behavior
