"""Zero-dependency Python code emitter for DecompiledProgram."""

from __future__ import annotations

from glassbox.ir import DecompiledProgram, RegimeLeaf, RegimeNode
from glassbox.types import TaskType

from .ir_emitter import (
    ExprDialect,
    collect_program_usage,
    format_float_literal,
    render_expr,
    render_guard,
    resolve_golden_vectors,
    sanitize_identifiers,
)

_PY_DIALECT = ExprDialect(
    abs_fn="abs",
    relu_fn="_gb_relu",
    tanh_fn="math.tanh",
    sin_fn="math.sin",
    cos_fn="math.cos",
    min_fn="min",
    max_fn="max",
    clamp_fn="_gb_clamp",
)


def _emit_py_tree(
    node: RegimeNode,
    feat_names: list[str],
    is_cls: bool,
    indent: int,
    lines: list[str],
) -> None:
    pad = " " * indent
    if isinstance(node, RegimeLeaf):
        lines.append(
            f"{pad}# Regime {node.regime_id}: {node.name} "
            f"(samples={node.sample_count}, local_err={node.local_error:.6g})"
        )
        exprs = [render_expr(expr, feat_names, _PY_DIALECT) for expr in node.output_exprs]
        lines.append(f"{pad}values = [{', '.join(exprs)}]")
        if is_cls:
            if node.class_id is not None:
                cls_expr = str(int(node.class_id))
            else:
                cls_expr = "_gb_argmax(values)"
        else:
            cls_expr = "None"
        lines.append(f"{pad}return values, {node.regime_id}, {cls_expr}")
        return

    cond = render_guard(node.guard, feat_names)
    lines.append(f"{pad}if {cond}:")
    _emit_py_tree(node.true_branch, feat_names, is_cls, indent + 4, lines)
    lines.append(f"{pad}else:")
    _emit_py_tree(node.false_branch, feat_names, is_cls, indent + 4, lines)


def emit_python(
    program: DecompiledProgram,
    golden_vectors: list[tuple[list[float], list[float], int]] | None = None,
) -> str:
    """Emit standalone zero-dependency Python code for a DecompiledProgram."""
    vectors = resolve_golden_vectors(program, golden_vectors)
    feat_names = sanitize_identifiers(program.domain.feature_names, prefix="x")
    out_names = sanitize_identifiers(program.domain.output_names, prefix="y")
    is_cls = program.task_type == TaskType.CLASSIFICATION
    leaves = program.leaves()

    used_vars, used_ops = collect_program_usage(program.root)
    needs_argmax = is_cls and any(leaf.class_id is None for leaf in leaves)

    lines = [
        f'"""Auto-generated standalone Python implementation for {program.name}."""',
        "",
        "from __future__ import annotations",
        "",
        "import math",
        "from dataclasses import dataclass",
        "",
    ]

    if "relu" in used_ops:
        lines.extend([
            "def _gb_relu(v: float) -> float:",
            "    return v if v > 0.0 else 0.0",
            "",
        ])
    if "clamp" in used_ops:
        lines.extend([
            "def _gb_clamp(v: float, lo: float, hi: float) -> float:",
            "    if v < lo:",
            "        return lo",
            "    if v > hi:",
            "        return hi",
            "    return v",
            "",
        ])
    if needs_argmax:
        lines.extend([
            "def _gb_argmax(vals: list[float]) -> int:",
            "    best_idx = 0",
            "    best_val = vals[0]",
            "    for idx in range(1, len(vals)):",
            "        if vals[idx] > best_val:",
            "            best_val = vals[idx]",
            "            best_idx = idx",
            "    return best_idx",
            "",
        ])

    lines.extend([
        "@dataclass(frozen=True)",
        "class GlassBoxOutput:",
    ])
    for oname in out_names:
        lines.append(f"    {oname}: float")
    lines.extend([
        "    regime_id: int",
        "    class_id: int | None = None",
        "",
        "    def to_list(self) -> list[float]:",
        f"        return [{', '.join(f'self.{oname}' for oname in out_names)}]",
        "",
        "",
        "def infer_with_regime(x: list[float] | tuple[float, ...]) -> tuple[list[float], int, int | None]:",
        f'    """Evaluate {program.name} and return (outputs, regime_id, class_id)."""',
    ])
    for idx in sorted(used_vars):
        lines.append(f"    {feat_names[idx]} = float(x[{idx}])")
    _emit_py_tree(program.root, feat_names, is_cls, 4, lines)

    lines.extend([
        "",
        "",
        "def infer(x: list[float] | tuple[float, ...]) -> list[float]:",
        f'    """Evaluate {program.name} and return output values."""',
        "    values, _, _ = infer_with_regime(x)",
        "    return values",
        "",
        "",
        "def infer_struct(x: list[float] | tuple[float, ...]) -> GlassBoxOutput:",
        f'    """Evaluate {program.name} and return a typed GlassBoxOutput."""',
        "    values, regime_id, class_id = infer_with_regime(x)",
        "    return GlassBoxOutput(",
    ])
    for j, oname in enumerate(out_names):
        lines.append(f"        {oname}=values[{j}],")
    lines.extend([
        "        regime_id=regime_id,",
        "        class_id=class_id,",
        "    )",
        "",
        "",
        'if __name__ == "__main__":',
    ])
    if not vectors:
        lines.append('    print("PASS: 0 golden vectors configured.")')
    else:
        lines.append("    _golden = [")
        for vx, vy, vrid in vectors:
            x_lit = ", ".join(format_float_literal(v) for v in vx)
            y_lit = ", ".join(format_float_literal(v) for v in vy)
            lines.append(f"        ([{x_lit}], [{y_lit}], {vrid}),")
        lines.extend([
            "    ]",
            "    for idx, (gx, gy, grid) in enumerate(_golden):",
            "        got_vals, got_reg, _ = infer_with_regime(gx)",
            "        assert got_reg == grid, f'vector {idx}: regime {got_reg} != {grid}'",
            "        for j, (got_v, exp_v) in enumerate(zip(got_vals, gy)):",
            "            assert math.fabs(got_v - exp_v) <= 1e-9, (",
            "                f'vector {idx} out[{j}]: {got_v} != {exp_v}'",
            "            )",
            '    print(f"PASS: {len(_golden)}/{len(_golden)} golden vectors verified.")',
        ])
    lines.append("")
    return "\n".join(lines)


__all__ = ["emit_python"]
