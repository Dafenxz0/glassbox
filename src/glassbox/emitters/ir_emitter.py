"""IR JSON, Mermaid flowchart, and shared emitter utilities."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from glassbox.ir import (
    BinaryExpr,
    ClampExpr,
    ConstExpr,
    DecompiledProgram,
    Expr,
    LinearGuard,
    RegimeLeaf,
    RegimeNode,
    UnaryExpr,
    VarExpr,
)

_C_RESERVED = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary",
}
_RUST_RESERVED = {
    "as", "break", "const", "continue", "crate", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
    "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
    "super", "trait", "true", "type", "unsafe", "use", "where", "while",
    "async", "await", "dyn", "abstract", "become", "box", "do", "final",
    "macro", "override", "priv", "typeof", "unsized", "virtual", "yield", "try",
}
_PY_RESERVED = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}
COMMON_RESERVED = (
    _C_RESERVED
    | _RUST_RESERVED
    | _PY_RESERVED
    | {"x", "out_values", "out_regime", "out_class", "regime_id", "class_id", "input", "out"}
)


@dataclass(frozen=True)
class ExprDialect:
    """Language-specific formatting callbacks for AST expressions."""

    abs_fn: str
    relu_fn: str
    tanh_fn: str
    sin_fn: str
    cos_fn: str
    min_fn: str
    max_fn: str
    clamp_fn: str
    method_unary: tuple[str, ...] = ()


def sanitize_identifiers(
    names: Sequence[str],
    *,
    reserved: set[str] | None = None,
    prefix: str = "v",
) -> list[str]:
    """Convert arbitrary feature/output names into unique, valid identifiers."""
    blocked = COMMON_RESERVED if reserved is None else (COMMON_RESERVED | reserved)
    used: set[str] = set()
    result: list[str] = []
    for idx, raw in enumerate(names):
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
        if not cleaned:
            cleaned = f"{prefix}{idx}"
        elif cleaned[0].isdigit():
            cleaned = f"{prefix}_{cleaned}"
        if cleaned in blocked:
            cleaned = f"{prefix}_{cleaned}"
        candidate = cleaned
        suffix = 1
        while candidate in used or candidate in blocked:
            candidate = f"{cleaned}_{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def sanitize_pascal_identifiers(
    names: Sequence[str],
    *,
    fallback_prefix: str = "Class",
) -> list[str]:
    """Convert class names into unique PascalCase Rust enum variant names."""
    used: set[str] = set()
    result: list[str] = []
    for idx, raw in enumerate(names):
        parts = [p for p in re.split(r"[^A-Za-z0-9]+", raw) if p]
        pascal = "".join(p[:1].upper() + p[1:] for p in parts)
        if not pascal or pascal[0].isdigit():
            pascal = f"{fallback_prefix}{idx}" if not pascal else f"{fallback_prefix}{pascal}"
        candidate = pascal
        suffix = 1
        while candidate in used:
            candidate = f"{pascal}{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def format_float_literal(value: float) -> str:
    """Format a finite float with a decimal point/exponent for C, Rust, and Python."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"Non-finite floating-point constant cannot be emitted: {v!r}")
    if v == 0.0:
        return "0.0"
    text = f"{v:.17g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _collect_expr_usage(expr: Expr, used_vars: set[int], used_ops: set[str]) -> None:
    if isinstance(expr, VarExpr):
        used_vars.add(expr.index)
    elif isinstance(expr, UnaryExpr):
        used_ops.add(expr.op)
        _collect_expr_usage(expr.operand, used_vars, used_ops)
    elif isinstance(expr, BinaryExpr):
        used_ops.add(expr.op)
        _collect_expr_usage(expr.left, used_vars, used_ops)
        _collect_expr_usage(expr.right, used_vars, used_ops)
    elif isinstance(expr, ClampExpr):
        used_ops.add("clamp")
        _collect_expr_usage(expr.operand, used_vars, used_ops)


def collect_program_usage(node: RegimeNode) -> tuple[set[int], set[str]]:
    """Collect all referenced feature indices and AST op names across a program tree."""
    used_vars: set[int] = set()
    used_ops: set[str] = set()

    def _walk(curr: RegimeNode) -> None:
        if isinstance(curr, RegimeLeaf):
            for expr in curr.output_exprs:
                _collect_expr_usage(expr, used_vars, used_ops)
            return
        for idx, w in enumerate(curr.guard.weights):
            if float(w) != 0.0:
                used_vars.add(idx)
        _walk(curr.true_branch)
        _walk(curr.false_branch)

    _walk(node)
    return used_vars, used_ops


def render_expr(expr: Expr, var_names: Sequence[str], dialect: ExprDialect) -> str:
    """Render an IR Expr AST into a target-language expression string."""
    if isinstance(expr, VarExpr):
        return var_names[expr.index]
    if isinstance(expr, ConstExpr):
        lit = format_float_literal(expr.value)
        return f"({lit})" if lit.startswith("-") else lit
    if isinstance(expr, UnaryExpr):
        inner = render_expr(expr.operand, var_names, dialect)
        if expr.op == "neg":
            return f"(-{inner})"
        if expr.op in dialect.method_unary:
            fn = getattr(dialect, f"{expr.op}_fn")
            return f"({inner}).{fn}()"
        fn_map = {
            "abs": dialect.abs_fn,
            "relu": dialect.relu_fn,
            "tanh": dialect.tanh_fn,
            "sin": dialect.sin_fn,
            "cos": dialect.cos_fn,
        }
        return f"{fn_map[expr.op]}({inner})"
    if isinstance(expr, BinaryExpr):
        left = render_expr(expr.left, var_names, dialect)
        right = render_expr(expr.right, var_names, dialect)
        op_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(expr.op)
        if op_sym is not None:
            return f"({left} {op_sym} {right})"
        fn_name = dialect.min_fn if expr.op == "min" else dialect.max_fn
        return f"{fn_name}({left}, {right})"
    if isinstance(expr, ClampExpr):
        inner = render_expr(expr.operand, var_names, dialect)
        lo = format_float_literal(expr.low)
        hi = format_float_literal(expr.high)
        return f"{dialect.clamp_fn}({inner}, {lo}, {hi})"
    raise TypeError(f"Unsupported expr type: {type(expr)!r}")


def render_guard(guard: LinearGuard, var_names: Sequence[str]) -> str:
    """Render a LinearGuard condition as a language-neutral float comparison."""
    terms: list[str] = []
    for idx, raw_w in enumerate(guard.weights):
        w = float(raw_w)
        if w == 0.0:
            continue
        var = var_names[idx]
        if not terms:
            if w == 1.0:
                terms.append(var)
            elif w == -1.0:
                terms.append(f"-{var}")
            else:
                terms.append(f"{format_float_literal(w)} * {var}")
        else:
            sign = " - " if w < 0.0 else " + "
            abs_w = abs(w)
            if abs_w == 1.0:
                terms.append(f"{sign}{var}")
            else:
                terms.append(f"{sign}{format_float_literal(abs_w)} * {var}")

    bias = float(guard.bias)
    if not terms:
        lhs = format_float_literal(bias)
    else:
        lhs = "".join(terms)
        if bias != 0.0:
            sign = " - " if bias < 0.0 else " + "
            lhs += f"{sign}{format_float_literal(abs(bias))}"
    return f"{lhs} {guard.op} 0.0"


def generate_golden_vectors(
    program: DecompiledProgram,
    count: int = 5,
) -> list[tuple[list[float], list[float], int]]:
    """Deterministically sample domain points and evaluate program golden vectors."""
    lows = program.domain.lower_bounds
    highs = program.domain.upper_bounds
    vectors: list[tuple[list[float], list[float], int]] = []
    for k in range(count):
        if k == 0:
            x = [0.5 * (lo + hi) for lo, hi in zip(lows, highs)]
        else:
            x = []
            for i, (lo, hi) in enumerate(zip(lows, highs)):
                frac = ((k * 0.6180339887498949 + (i + 1) * 0.3247179572447460) % 0.94) + 0.03
                x.append(lo + frac * (hi - lo))
        outputs, regime_id = program.evaluate_one(x)
        vectors.append((x, outputs, regime_id))
    return vectors


def resolve_golden_vectors(
    program: DecompiledProgram,
    golden_vectors: Sequence[tuple[list[float], list[float], int]] | None,
) -> list[tuple[list[float], list[float], int]]:
    """Return validated golden vectors, auto-generating 5 deterministic ones if None."""
    if golden_vectors is None:
        return generate_golden_vectors(program, count=5)
    in_dim = program.domain.input_dim
    out_dim = program.domain.output_dim
    resolved: list[tuple[list[float], list[float], int]] = []
    for idx, (x, y, rid) in enumerate(golden_vectors):
        if len(x) != in_dim:
            raise ValueError(f"golden_vectors[{idx}] input dim {len(x)} != {in_dim}")
        if len(y) != out_dim:
            raise ValueError(f"golden_vectors[{idx}] output dim {len(y)} != {out_dim}")
        resolved.append(([float(v) for v in x], [float(v) for v in y], int(rid)))
    return resolved


def _escape_mermaid_label(text: str) -> str:
    """Sanitize a Mermaid node label so it has no unescaped quotes or HTML tags."""
    cleaned = text.replace('"', "'").replace("\n", " ").replace("\r", " ")
    # Strip any HTML-like tags if user-supplied names/descriptions contain them.
    cleaned = re.sub(r"</?[A-Za-z][^>]*>", "", cleaned)
    return cleaned.strip()


def emit_ir_json(program: DecompiledProgram) -> str:
    """Emit indented canonical JSON representation of a DecompiledProgram."""
    return json.dumps(program.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def emit_mermaid(program: DecompiledProgram) -> str:
    """Emit a GitHub-safe Mermaid flowchart TD diagram of the regime guard tree."""
    lines = ["flowchart TD"]
    guard_counter = 0
    leaf_counter = 0

    def _walk(node: RegimeNode) -> str:
        nonlocal guard_counter, leaf_counter
        if isinstance(node, RegimeLeaf):
            node_id = f"l{leaf_counter}"
            leaf_counter += 1
            cls_part = f" | class={node.class_name or node.class_id}" if node.class_id is not None else ""
            label = _escape_mermaid_label(
                f"Regime {node.regime_id}: {node.name}"
                f" (n={node.sample_count}, err={node.local_error:.4f}{cls_part})"
            )
            lines.append(f'    {node_id}["{label}"]')
            return node_id

        node_id = f"n{guard_counter}"
        guard_counter += 1
        guard_label = _escape_mermaid_label(f"{node.guard.to_readable(precision=4)}?")
        lines.append(f'    {node_id}["{guard_label}"]')
        true_id = _walk(node.true_branch)
        false_id = _walk(node.false_branch)
        lines.append(f'    {node_id} -->|"true"| {true_id}')
        lines.append(f'    {node_id} -->|"false"| {false_id}')
        return node_id

    _walk(program.root)
    return "\n".join(lines) + "\n"


__all__ = [
    "ExprDialect",
    "collect_program_usage",
    "emit_ir_json",
    "emit_mermaid",
    "format_float_literal",
    "generate_golden_vectors",
    "render_expr",
    "render_guard",
    "resolve_golden_vectors",
    "sanitize_identifiers",
    "sanitize_pascal_identifiers",
]
