"""Zero-dependency Rust (no_std) code emitter for DecompiledProgram."""

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
    sanitize_pascal_identifiers,
)

_RUST_DIALECT = ExprDialect(
    abs_fn="gb_abs",
    relu_fn="gb_relu",
    tanh_fn="tanh",
    sin_fn="sin",
    cos_fn="cos",
    min_fn="gb_min",
    max_fn="gb_max",
    clamp_fn="gb_clamp",
    method_unary=("tanh", "sin", "cos"),
)


def _emit_rust_tree(
    node: RegimeNode,
    feat_names: list[str],
    out_names: list[str],
    is_cls: bool,
    class_variants: list[str],
    indent: int,
    lines: list[str],
) -> None:
    pad = " " * indent
    if isinstance(node, RegimeLeaf):
        lines.append(
            f"{pad}// Regime {node.regime_id}: {node.name} "
            f"(samples={node.sample_count}, local_err={node.local_error:.6g})"
        )
        for idx, expr in enumerate(node.output_exprs):
            rendered = render_expr(expr, feat_names, _RUST_DIALECT, group=False)
            lines.append(f"{pad}let out_{idx}: f64 = {rendered};")
        if is_cls:
            if node.class_id is not None:
                cls_expr = f"ClassId::{class_variants[int(node.class_id)]}"
            else:
                args = ", ".join(f"out_{idx}" for idx in range(len(out_names)))
                cls_expr = f"ClassId::from_index(gb_argmax(&[{args}]))"
        lines.append(f"{pad}Output {{")
        for idx, oname in enumerate(out_names):
            lines.append(f"{pad}    {oname}: out_{idx},")
        lines.append(f"{pad}    regime_id: {node.regime_id},")
        if is_cls:
            lines.append(f"{pad}    class_id: {cls_expr},")
        lines.append(f"{pad}}}")
        return

    cond = render_guard(node.guard, feat_names)
    lines.append(f"{pad}if {cond} {{")
    _emit_rust_tree(
        node.true_branch,
        feat_names,
        out_names,
        is_cls,
        class_variants,
        indent + 4,
        lines,
    )
    lines.append(f"{pad}}} else {{")
    _emit_rust_tree(
        node.false_branch,
        feat_names,
        out_names,
        is_cls,
        class_variants,
        indent + 4,
        lines,
    )
    lines.append(f"{pad}}}")


def emit_rust(
    program: DecompiledProgram,
    golden_vectors: list[tuple[list[float], list[float], int]] | None = None,
) -> str:
    """Emit standalone `#![cfg_attr(not(test), no_std)]` Rust code for a DecompiledProgram."""
    vectors = resolve_golden_vectors(program, golden_vectors)
    feat_names = sanitize_identifiers(program.domain.feature_names, prefix="x")
    out_names = sanitize_identifiers(program.domain.output_names, prefix="y")
    in_dim = program.domain.input_dim
    is_cls = program.task_type == TaskType.CLASSIFICATION
    leaves = program.leaves()

    class_variants: list[str] = []
    if is_cls:
        max_leaf_cls = max(
            (int(leaf.class_id) + 1 for leaf in leaves if leaf.class_id is not None),
            default=0,
        )
        num_classes = max(len(program.domain.class_names), program.domain.output_dim, max_leaf_cls)
        raw_cls = list(program.domain.class_names)
        while len(raw_cls) < num_classes:
            raw_cls.append(f"Class{len(raw_cls)}")
        class_variants = sanitize_pascal_identifiers(raw_cls)

    used_vars, used_ops = collect_program_usage(program.root)
    needs_argmax = is_cls and any(leaf.class_id is None for leaf in leaves)

    lines = [
        f"// Auto-generated Rust implementation by GlassBox for {program.name}",
        *([] if used_ops & {"tanh", "sin", "cos"} else ["#![cfg_attr(not(test), no_std)]"]),
        "",
    ]

    if "abs" in used_ops:
        lines.extend([
            "#[inline]",
            "fn gb_abs(v: f64) -> f64 {",
            "    if v < 0.0 { -v } else { v }",
            "}",
            "",
        ])
    if "relu" in used_ops:
        lines.extend([
            "#[inline]",
            "fn gb_relu(v: f64) -> f64 {",
            "    if v > 0.0 { v } else { 0.0 }",
            "}",
            "",
        ])
    if "min" in used_ops:
        lines.extend([
            "#[inline]",
            "fn gb_min(a: f64, b: f64) -> f64 {",
            "    if a < b { a } else { b }",
            "}",
            "",
        ])
    if "max" in used_ops:
        lines.extend([
            "#[inline]",
            "fn gb_max(a: f64, b: f64) -> f64 {",
            "    if a > b { a } else { b }",
            "}",
            "",
        ])
    if "clamp" in used_ops:
        lines.extend([
            "#[inline]",
            "fn gb_clamp(v: f64, lo: f64, hi: f64) -> f64 {",
            "    if v < lo { lo } else if v > hi { hi } else { v }",
            "}",
            "",
        ])
    if needs_argmax:
        lines.extend([
            "#[inline]",
            "fn gb_argmax(vals: &[f64]) -> usize {",
            "    let mut best_idx: usize = 0;",
            "    let mut best_val: f64 = vals[0];",
            "    let mut i: usize = 1;",
            "    while i < vals.len() {",
            "        if vals[i] > best_val {",
            "            best_val = vals[i];",
            "            best_idx = i;",
            "        }",
            "        i += 1;",
            "    }",
            "    best_idx",
            "}",
            "",
        ])

    if is_cls:
        lines.extend([
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
            "#[allow(dead_code)]",
            "pub enum ClassId {",
        ])
        for idx, variant in enumerate(class_variants):
            lines.append(f"    {variant} = {idx},")
        lines.extend([
            "}",
            "",
            "impl ClassId {",
            "    #[allow(dead_code)]",
            "    pub const fn from_index(idx: usize) -> Self {",
            "        match idx {",
        ])
        for idx, variant in enumerate(class_variants):
            lines.append(f"            {idx} => Self::{variant},")
        lines.extend([
            f"            _ => Self::{class_variants[0]},",
            "        }",
            "    }",
            "",
            "    #[allow(dead_code)]",
            "    pub const fn as_usize(self) -> usize {",
            "        self as usize",
            "    }",
            "}",
            "",
        ])

    lines.extend([
        "#[derive(Debug, Clone, Copy)]",
        "pub struct Input {",
    ])
    for fname in feat_names:
        lines.append(f"    pub {fname}: f64,")
    lines.extend([
        "}",
        "",
        "#[derive(Debug, Clone, Copy)]",
        "pub struct Output {",
    ])
    for oname in out_names:
        lines.append(f"    pub {oname}: f64,")
    lines.append("    pub regime_id: usize,")
    if is_cls:
        lines.append("    pub class_id: ClassId,")
    lines.extend([
        "}",
        "",
        f"/// Evaluate decompiled `{program.name}` across {len(leaves)} regimes.",
    ])
    for leaf in leaves:
        lines.append(
            f"/// - Regime {leaf.regime_id} (`{leaf.name}`): "
            f"samples={leaf.sample_count}, local_err={leaf.local_error:.6g}"
        )
    lines.append("pub fn infer(input: &Input) -> Output {")
    for idx in range(in_dim):
        fname = feat_names[idx]
        if idx in used_vars:
            lines.append(f"    let {fname}: f64 = input.{fname};")
        else:
            lines.append(f"    let _ = input.{fname};")
    _emit_rust_tree(program.root, feat_names, out_names, is_cls, class_variants, 4, lines)
    lines.extend([
        "}",
        "",
        "#[cfg(test)]",
        "mod tests {",
        "    use super::*;",
        "",
        "    #[test]",
        "    fn test_golden_vectors() {",
    ])
    if not vectors:
        mids = [
            format_float_literal(0.5 * (lo + hi))
            for lo, hi in zip(program.domain.lower_bounds, program.domain.upper_bounds)
        ]
        lines.append("        let inp = Input {")
        for fname, mid in zip(feat_names, mids):
            lines.append(f"            {fname}: {mid},")
        lines.extend([
            "        };",
            "        let out = infer(&inp);",
            "        assert!(out.regime_id < usize::MAX);",
        ])
    else:
        for idx, (vx, vy, vrid) in enumerate(vectors):
            lines.append(f"        let inp_{idx} = Input {{")
            for fname, val in zip(feat_names, vx):
                lines.append(f"            {fname}: {format_float_literal(val)},")
            lines.extend([
                "        };",
                f"        let out_{idx} = infer(&inp_{idx});",
                f"        assert_eq!(out_{idx}.regime_id, {vrid});",
            ])
            for oname, exp in zip(out_names, vy):
                lines.append(
                    f"        assert!((out_{idx}.{oname} - {format_float_literal(exp)}).abs() < 1e-9);"
                )
            if is_cls:
                lines.append(f"        let _ = out_{idx}.class_id.as_usize();")
    lines.extend([
        "    }",
        "}",
        "",
    ])
    return "\n".join(lines)


__all__ = ["emit_rust"]
