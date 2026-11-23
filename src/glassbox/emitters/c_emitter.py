"""Zero-dependency C99 code emitter and native compiler helpers for DecompiledProgram."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

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

_C_DIALECT = ExprDialect(
    abs_fn="fabs",
    relu_fn="GB_RELU",
    tanh_fn="tanh",
    sin_fn="sin",
    cos_fn="cos",
    min_fn="fmin",
    max_fn="fmax",
    clamp_fn="GB_CLAMP",
)


def find_c_compiler() -> tuple[list[str], str] | None:
    """Locate an available C99 compiler toolchain (`gcc`, `clang`, or `ziglang cc`)."""
    gcc = shutil.which("gcc")
    if gcc:
        return ([gcc], "gcc")
    clang = shutil.which("clang")
    if clang:
        return ([clang], "clang")
    if importlib.util.find_spec("ziglang") is not None:
        return ([sys.executable, "-m", "ziglang", "cc"], "ziglang cc")
    return None


def _write_companion_headers(build_dir: Path, stem: str, c_source: str, h_source: str | None) -> None:
    if not h_source or not h_source.strip():
        return
    (build_dir / f"{stem}.h").write_text(h_source, encoding="utf-8")
    (build_dir / "glassbox_model.h").write_text(h_source, encoding="utf-8")
    inc_match = re.search(r'#include\s+"([^"]+)"', c_source)
    if inc_match:
        (build_dir / Path(inc_match.group(1)).name).write_text(h_source, encoding="utf-8")


def compile_c_shared(
    c_source: str,
    out_path: Path | str,
    h_source: str | None = None,
    *,
    strict_warnings: bool = True,
    extra_flags: Sequence[str] = (),
) -> tuple[Path, str]:
    """Compile emitted C99 source into a native shared library (.dll / .so)."""
    compiler = find_c_compiler()
    if compiler is None:
        raise RuntimeError("No C99 compiler available (checked gcc, clang, ziglang)")
    cc_cmd, cc_name = compiler

    ext = ".dll" if sys.platform == "win32" else ".so"
    target = Path(out_path)
    if target.exists() and target.is_dir():
        lib_path = target / f"glassbox_model{ext}"
    elif target.suffix not in {".dll", ".so", ".dylib"}:
        lib_path = target.with_suffix(ext)
    else:
        lib_path = target

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    c_path = lib_path.with_suffix(".c")
    c_path.write_text(c_source, encoding="utf-8")
    _write_companion_headers(lib_path.parent, lib_path.stem, c_source, h_source)

    cmd = [*cc_cmd, "-std=c99", "-O3"]
    if strict_warnings:
        cmd.extend(["-Wall", "-Wextra", "-Werror"])
    cmd.append("-shared")
    if sys.platform != "win32":
        cmd.append("-fPIC")
    cmd.extend([*extra_flags, str(c_path), "-o", str(lib_path), "-lm"])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0 or not lib_path.exists():
        err_msg = (proc.stderr or proc.stdout).strip()[:400]
        raise RuntimeError(f"C99 shared compile failed ({cc_name}): {err_msg}")
    return lib_path, cc_name


def compile_c_executable(
    c_source: str,
    out_path: Path | str,
    h_source: str | None = None,
    *,
    define_main: bool = True,
    strict_warnings: bool = True,
    extra_flags: Sequence[str] = (),
) -> tuple[Path, str]:
    """Compile emitted C99 source into a standalone executable (with `-DGLASSBOX_MAIN`)."""
    compiler = find_c_compiler()
    if compiler is None:
        raise RuntimeError("No C99 compiler available (checked gcc, clang, ziglang)")
    cc_cmd, cc_name = compiler

    ext = ".exe" if sys.platform == "win32" else ""
    target = Path(out_path)
    if target.exists() and target.is_dir():
        exe_path = target / f"glassbox_main{ext}"
    elif sys.platform == "win32" and target.suffix.lower() != ".exe":
        exe_path = target.with_suffix(".exe")
    else:
        exe_path = target

    exe_path.parent.mkdir(parents=True, exist_ok=True)
    c_path = exe_path.parent / f"{exe_path.stem}_main.c"
    c_path.write_text(c_source, encoding="utf-8")
    _write_companion_headers(exe_path.parent, exe_path.stem, c_source, h_source)

    cmd = [*cc_cmd, "-std=c99", "-O3"]
    if strict_warnings:
        cmd.extend(["-Wall", "-Wextra", "-Werror"])
    if define_main:
        cmd.append("-DGLASSBOX_MAIN")
    cmd.extend([*extra_flags, str(c_path), "-o", str(exe_path), "-lm"])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0 or not exe_path.exists():
        err_msg = (proc.stderr or proc.stdout).strip()[:400]
        raise RuntimeError(f"C99 executable compile failed ({cc_name}): {err_msg}")
    return exe_path, cc_name


def _safe_comment(text: str) -> str:
    return text.replace("*/", "* /").replace("\n", " ").replace("\r", " ").strip()


def _header_guard(program_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", program_name).strip("_").upper() or "PROGRAM"
    return f"GLASSBOX_{cleaned}_H"


def _shared_header_decls(
    feat_names: list[str],
    out_names: list[str],
    in_dim: int,
    out_dim: int,
) -> list[str]:
    lines = [
        "#ifdef _WIN32",
        "#define GLASSBOX_API __declspec(dllexport)",
        "#else",
        "#define GLASSBOX_API",
        "#endif",
        "",
        f"#define GLASSBOX_INPUT_DIM {in_dim}",
        f"#define GLASSBOX_OUTPUT_DIM {out_dim}",
        "",
        "typedef struct {",
    ]
    for name in feat_names:
        lines.append(f"    double {name};")
    lines.extend([
        "} GlassBoxInput;",
        "",
        "typedef struct {",
    ])
    for name in out_names:
        lines.append(f"    double {name};")
    lines.extend([
        "    int regime_id;",
        "    int class_id;",
        "} GlassBoxOutput;",
        "",
        "GLASSBOX_API int glassbox_infer(const double* x, double* out_values, int* out_regime);",
        "GLASSBOX_API GlassBoxOutput glassbox_infer_struct(const GlassBoxInput* in);",
        "GLASSBOX_API double glassbox_bench_loop(const double* xs, int n_samples, int n_repeats, double* out_last);",
    ])
    return lines


def _emit_c_tree(
    node: RegimeNode,
    feat_names: list[str],
    is_cls: bool,
    out_dim: int,
    indent: int,
    lines: list[str],
) -> None:
    pad = " " * indent
    if isinstance(node, RegimeLeaf):
        comment = _safe_comment(
            f"Regime {node.regime_id}: {node.name} "
            f"(samples={node.sample_count}, local_err={node.local_error:.6g})"
        )
        lines.append(f"{pad}/* {comment} */")
        for idx, expr in enumerate(node.output_exprs):
            rendered = render_expr(expr, feat_names, _C_DIALECT)
            lines.append(f"{pad}out_values[{idx}] = {rendered};")
        lines.append(f"{pad}regime_id = {node.regime_id};")
        if is_cls:
            if node.class_id is not None:
                lines.append(f"{pad}class_id = {int(node.class_id)};")
            else:
                lines.append(f"{pad}class_id = gb_argmax(out_values, {out_dim});")
        else:
            lines.append(f"{pad}class_id = -1;")
        return

    cond = render_guard(node.guard, feat_names)
    lines.append(f"{pad}if ({cond}) {{")
    _emit_c_tree(node.true_branch, feat_names, is_cls, out_dim, indent + 4, lines)
    lines.append(f"{pad}}} else {{")
    _emit_c_tree(node.false_branch, feat_names, is_cls, out_dim, indent + 4, lines)
    lines.append(f"{pad}}}")


def emit_c(
    program: DecompiledProgram,
    golden_vectors: list[tuple[list[float], list[float], int]] | None = None,
) -> tuple[str, str]:
    """Emit standalone C99 source and header files for a DecompiledProgram."""
    vectors = resolve_golden_vectors(program, golden_vectors)
    feat_names = sanitize_identifiers(program.domain.feature_names, prefix="x")
    out_names = sanitize_identifiers(program.domain.output_names, prefix="y")
    in_dim = program.domain.input_dim
    out_dim = program.domain.output_dim
    is_cls = program.task_type == TaskType.CLASSIFICATION

    guard_macro = _header_guard(program.name)
    decl_lines = _shared_header_decls(feat_names, out_names, in_dim, out_dim)

    h_lines = [
        f"/* Auto-generated by GlassBox for {_safe_comment(program.name)} */",
        f"#ifndef {guard_macro}",
        f"#define {guard_macro}",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        *decl_lines,
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        f"#endif /* {guard_macro} */",
        "",
    ]

    used_vars, used_ops = collect_program_usage(program.root)
    leaves = program.leaves()
    needs_argmax = is_cls and any(leaf.class_id is None for leaf in leaves)

    c_lines = [
        f"/* Auto-generated C99 implementation by GlassBox for {_safe_comment(program.name)} */",
        "#include <math.h>",
        "#include <stddef.h>",
        "#ifdef GLASSBOX_MAIN",
        "#include <stdio.h>",
        "#endif",
        "",
        f"#ifndef {guard_macro}",
        f"#define {guard_macro}",
        *decl_lines,
        "#endif",
        "",
    ]

    if "relu" in used_ops:
        c_lines.extend([
            "static inline double gb_relu(double v) {",
            "    return v > 0.0 ? v : 0.0;",
            "}",
            "#define GB_RELU(v) gb_relu(v)",
            "",
        ])
    if "clamp" in used_ops:
        c_lines.extend([
            "static inline double gb_clamp(double v, double lo, double hi) {",
            "    return v < lo ? lo : (v > hi ? hi : v);",
            "}",
            "#define GB_CLAMP(v, lo, hi) gb_clamp(v, lo, hi)",
            "",
        ])
    if needs_argmax:
        c_lines.extend([
            "static inline int gb_argmax(const double* vals, int n) {",
            "    int best_idx = 0;",
            "    double best_val = vals[0];",
            "    for (int i = 1; i < n; ++i) {",
            "        if (vals[i] > best_val) {",
            "            best_val = vals[i];",
            "            best_idx = i;",
            "        }",
            "    }",
            "    return best_idx;",
            "}",
            "",
        ])

    c_lines.extend([
        "static void glassbox_eval_core(",
        "    const double* x,",
        "    double* out_values,",
        "    int* out_regime,",
        "    int* out_class",
        ") {",
    ])
    if not used_vars:
        c_lines.append("    (void)x;")
    for idx in sorted(used_vars):
        c_lines.append(f"    const double {feat_names[idx]} = x[{idx}];")
    c_lines.extend([
        "    int regime_id = 0;",
        "    int class_id = -1;",
        "",
    ])
    _emit_c_tree(program.root, feat_names, is_cls, out_dim, 4, c_lines)
    c_lines.extend([
        "",
        "    if (out_regime != NULL) {",
        "        *out_regime = regime_id;",
        "    }",
        "    if (out_class != NULL) {",
        "        *out_class = class_id;",
        "    }",
        "}",
        "",
        "GLASSBOX_API int glassbox_infer(const double* x, double* out_values, int* out_regime) {",
        "    if (x == NULL || out_values == NULL) {",
        "        return -1;",
        "    }",
        "    glassbox_eval_core(x, out_values, out_regime, NULL);",
        "    return 0;",
        "}",
        "",
        "GLASSBOX_API GlassBoxOutput glassbox_infer_struct(const GlassBoxInput* in) {",
        "    GlassBoxOutput result = {0};",
        "    if (in == NULL) {",
        "        result.regime_id = -1;",
        "        result.class_id = -1;",
        "        return result;",
        "    }",
        f"    const double x[{in_dim}] = {{",
    ])
    for i, fname in enumerate(feat_names):
        comma = "," if i + 1 < in_dim else ""
        c_lines.append(f"        in->{fname}{comma}")
    c_lines.extend([
        "    };",
        f"    double out_vals[{out_dim}];",
        "    glassbox_eval_core(x, out_vals, &result.regime_id, &result.class_id);",
    ])
    for j, oname in enumerate(out_names):
        c_lines.append(f"    result.{oname} = out_vals[{j}];")
    c_lines.extend([
        "    return result;",
        "}",
        "",
        "GLASSBOX_API double glassbox_bench_loop(",
        "    const double* xs,",
        "    int n_samples,",
        "    int n_repeats,",
        "    double* out_last",
        ") {",
        "    if (xs == NULL || n_samples <= 0 || n_repeats <= 0) {",
        "        return 0.0;",
        "    }",
        "    double local_out[GLASSBOX_OUTPUT_DIM] = {0};",
        "    int local_regime = 0;",
        "    volatile double acc = 0.0;",
        "    for (int r = 0; r < n_repeats; ++r) {",
        "        for (int i = 0; i < n_samples; ++i) {",
        "            const double* row = xs + (size_t)i * GLASSBOX_INPUT_DIM;",
        "            glassbox_eval_core(row, local_out, &local_regime, NULL);",
        "            acc = acc + local_out[0] + (double)local_regime * 1e-12;",
        "        }",
        "    }",
        "    if (out_last != NULL) {",
        "        for (int j = 0; j < GLASSBOX_OUTPUT_DIM; ++j) {",
        "            out_last[j] = local_out[j];",
        "        }",
        "    }",
        "    return acc;",
        "}",
        "",
        "#ifdef GLASSBOX_MAIN",
        "int main(void) {",
    ])

    if not vectors:
        c_lines.extend([
            '    printf("GlassBox C99 self-check: 0 golden vectors.\\n");',
            "    return 0;",
        ])
    else:
        n_vec = len(vectors)
        c_lines.append(f"    const double gv_x[{n_vec}][{in_dim}] = {{")
        for i, (vx, _, _) in enumerate(vectors):
            row = ", ".join(format_float_literal(v) for v in vx)
            comma = "," if i + 1 < n_vec else ""
            c_lines.append(f"        {{{row}}}{comma}")
        c_lines.append("    };")

        c_lines.append(f"    const double gv_y[{n_vec}][{out_dim}] = {{")
        for i, (_, vy, _) in enumerate(vectors):
            row = ", ".join(format_float_literal(v) for v in vy)
            comma = "," if i + 1 < n_vec else ""
            c_lines.append(f"        {{{row}}}{comma}")
        c_lines.append("    };")

        regimes_str = ", ".join(str(rid) for _, _, rid in vectors)
        c_lines.extend([
            f"    const int gv_regime[{n_vec}] = {{{regimes_str}}};",
            f"    for (int i = 0; i < {n_vec}; ++i) {{",
            f"        double out[{out_dim}];",
            "        int regime = -1;",
            "        if (glassbox_infer(gv_x[i], out, &regime) != 0) {",
            '            fprintf(stderr, "FAIL vector %d: glassbox_infer returned non-zero\\n", i);',
            "            return 1;",
            "        }",
            "        if (regime != gv_regime[i]) {",
            '            fprintf(stderr, "FAIL vector %d: regime %d != %d\\n", i, regime, gv_regime[i]);',
            "            return 1;",
            "        }",
            f"        for (int j = 0; j < {out_dim}; ++j) {{",
            "            if (fabs(out[j] - gv_y[i][j]) > 1e-9) {",
            '                fprintf(stderr, "FAIL vector %d out[%d]: %.17g != %.17g\\n", i, j, out[j], gv_y[i][j]);',
            "                return 1;",
            "            }",
            "        }",
            "        GlassBoxInput s_in;",
        ])
        for idx, fname in enumerate(feat_names):
            c_lines.append(f"        s_in.{fname} = gv_x[i][{idx}];")
        c_lines.extend([
            "        GlassBoxOutput s_out = glassbox_infer_struct(&s_in);",
            "        if (s_out.regime_id != gv_regime[i]) {",
            '            fprintf(stderr, "FAIL struct vector %d: regime %d != %d\\n", i, s_out.regime_id, gv_regime[i]);',
            "            return 1;",
            "        }",
        ])
        for j, oname in enumerate(out_names):
            c_lines.extend([
                f"        if (fabs(s_out.{oname} - gv_y[i][{j}]) > 1e-9) {{",
                f'            fprintf(stderr, "FAIL struct vector %d {oname}: %.17g != %.17g\\n", i, s_out.{oname}, gv_y[i][{j}]);',
                "            return 1;",
                "        }",
            ])
        c_lines.extend([
            "    }",
            f"    double bench_last[{out_dim}] = {{0}};",
            f"    (void)glassbox_bench_loop(&gv_x[0][0], {n_vec}, 2, bench_last);",
            f"    for (int j = 0; j < {out_dim}; ++j) {{",
            f"        if (fabs(bench_last[j] - gv_y[{n_vec - 1}][j]) > 1e-9) {{",
            '            fprintf(stderr, "FAIL bench_loop out[%d]\\n", j);',
            "            return 1;",
            "        }",
            "    }",
            f'    printf("PASS: %d/%d golden vectors verified.\\n", {n_vec}, {n_vec});',
            "    return 0;",
        ])

    c_lines.extend([
        "}",
        "#endif /* GLASSBOX_MAIN */",
        "",
    ])

    return "\n".join(c_lines), "\n".join(h_lines)


__all__ = [
    "compile_c_executable",
    "compile_c_shared",
    "emit_c",
    "find_c_compiler",
]
