"""Code and diagram emitters for DecompiledProgram."""

from __future__ import annotations

from glassbox.ir import DecompiledProgram

from .c_emitter import (
    compile_c_executable,
    compile_c_shared,
    emit_c,
    find_c_compiler,
)
from .ir_emitter import (
    emit_ir_json,
    emit_mermaid,
    generate_golden_vectors,
    resolve_golden_vectors,
)
from .python_emitter import emit_python
from .rust_emitter import emit_rust


def emit_all(
    program: DecompiledProgram,
    golden_vectors: list[tuple[list[float], list[float], int]] | None = None,
) -> dict[str, str]:
    """Emit all target artifacts (C99, header, Rust, Python, Mermaid, IR JSON)."""
    vectors = resolve_golden_vectors(program, golden_vectors)
    c_source, h_source = emit_c(program, golden_vectors=vectors)
    rust_source = emit_rust(program, golden_vectors=vectors)
    python_source = emit_python(program, golden_vectors=vectors)
    mermaid_source = emit_mermaid(program)
    ir_json_source = emit_ir_json(program)
    return {
        "c": c_source,
        "h": h_source,
        "rust": rust_source,
        "python": python_source,
        "mermaid": mermaid_source,
        "ir_json": ir_json_source,
    }


__all__ = [
    "compile_c_executable",
    "compile_c_shared",
    "emit_all",
    "emit_c",
    "emit_ir_json",
    "emit_mermaid",
    "emit_python",
    "emit_rust",
    "find_c_compiler",
    "generate_golden_vectors",
]
