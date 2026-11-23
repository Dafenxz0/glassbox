"""Unit and native compiler integration tests for Glassbox code emitters."""

from __future__ import annotations

import ast
import ctypes
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from glassbox.emitters import (
    compile_c_executable,
    compile_c_shared,
    emit_all,
    emit_c,
    emit_ir_json,
    emit_mermaid,
    emit_python,
    emit_rust,
    find_c_compiler,
    generate_golden_vectors,
)
from glassbox.emitters.ir_emitter import sanitize_identifiers
from glassbox.ir import (
    BinaryExpr,
    ClampExpr,
    ConstExpr,
    DecompiledProgram,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    UnaryExpr,
    VarExpr,
)
from glassbox.types import DomainSpec, TaskType

if sys.platform == "win32":
    import _ctypes


def _make_rich_program(task_type: TaskType = TaskType.REGRESSION) -> DecompiledProgram:
    domain = DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -1.5],
        upper_bounds=[2.0, 2.5],
        output_names=["y0", "y1"],
        task_type=task_type,
        class_names=["stay", "boost"] if task_type == TaskType.CLASSIFICATION else [],
    )
    x0 = VarExpr(index=0, name="x0")
    x1 = VarExpr(index=1, name="x1")

    leaf0 = RegimeLeaf(
        regime_id=0,
        name="left_low",
        description="x0 <= 0 and x1 < 1",
        output_exprs=[
            BinaryExpr(
                op="add",
                left=BinaryExpr(op="mul", left=ConstExpr(value=2.0), right=x0),
                right=ConstExpr(value=1.0),
            ),
            ClampExpr(operand=x1, low=-0.5, high=1.5),
        ],
        class_id=0 if task_type == TaskType.CLASSIFICATION else None,
        class_name="stay" if task_type == TaskType.CLASSIFICATION else None,
        sample_count=50,
        local_error=0.01,
    )
    leaf1 = RegimeLeaf(
        regime_id=1,
        name="left_high",
        description="x0 <= 0 and x1 >= 1",
        output_exprs=[
            UnaryExpr(op="relu", operand=x1),
            UnaryExpr(op="abs", operand=x0),
        ],
        class_id=1 if task_type == TaskType.CLASSIFICATION else None,
        class_name="boost" if task_type == TaskType.CLASSIFICATION else None,
        sample_count=30,
        local_error=0.02,
    )
    leaf2 = RegimeLeaf(
        regime_id=2,
        name="right",
        description="x0 > 0",
        output_exprs=[
            BinaryExpr(op="min", left=x0, right=x1),
            BinaryExpr(
                op="max",
                left=UnaryExpr(op="neg", operand=x0),
                right=BinaryExpr(op="div", left=x1, right=ConstExpr(value=2.0)),
            ),
        ],
        class_id=None,
        sample_count=40,
        local_error=0.005,
    )
    inner = GuardNode(
        guard=LinearGuard(weights=[0.0, 1.0], bias=-1.0, feature_names=["x0", "x1"], op="<"),
        true_branch=leaf0,
        false_branch=leaf1,
    )
    root = GuardNode(
        guard=LinearGuard(weights=[1.0, 0.0], bias=0.0, feature_names=["x0", "x1"], op="<="),
        true_branch=inner,
        false_branch=leaf2,
    )
    return DecompiledProgram(
        name="rich_test_prog",
        task_type=task_type,
        domain=domain,
        root=root,
    )


def _make_constant_program() -> DecompiledProgram:
    """Single-leaf constant program to ensure zero unused-parameter/variable warnings."""
    domain = DomainSpec(
        feature_names=["in", "1st val"],
        lower_bounds=[0.0, -1.0],
        upper_bounds=[1.0, 1.0],
        output_names=["class", "regime_id"],
        task_type=TaskType.REGRESSION,
    )
    leaf = RegimeLeaf(
        regime_id=0,
        name="const_only",
        description="constant leaf",
        output_exprs=[ConstExpr(value=3.25), ConstExpr(value=-1.5)],
        sample_count=10,
        local_error=0.0,
    )
    return DecompiledProgram(
        name="const-prog 01",
        task_type=TaskType.REGRESSION,
        domain=domain,
        root=leaf,
    )


def _make_nonlinear_program() -> DecompiledProgram:
    domain = DomainSpec(
        feature_names=["angle", "vel"],
        lower_bounds=[-1.0, -2.0],
        upper_bounds=[1.0, 2.0],
        output_names=["u0"],
        task_type=TaskType.REGRESSION,
    )
    a = VarExpr(index=0, name="angle")
    v = VarExpr(index=1, name="vel")
    leaf = RegimeLeaf(
        regime_id=0,
        name='quote"name<br>',
        description="smooth",
        output_exprs=[
            BinaryExpr(
                op="add",
                left=UnaryExpr(op="tanh", operand=a),
                right=BinaryExpr(
                    op="sub",
                    left=UnaryExpr(op="sin", operand=v),
                    right=UnaryExpr(op="cos", operand=a),
                ),
            )
        ],
        sample_count=20,
        local_error=0.001,
    )
    return DecompiledProgram(
        name="smooth_ctrl",
        task_type=TaskType.REGRESSION,
        domain=domain,
        root=leaf,
        is_linear_ir=False,
    )


def test_find_c_compiler_available() -> None:
    found = find_c_compiler()
    assert found is not None
    cmd, name = found
    assert isinstance(cmd, list) and len(cmd) >= 1
    assert name in {"gcc", "clang", "ziglang cc"}


@pytest.mark.parametrize(
    "program",
    [
        _make_rich_program(TaskType.REGRESSION),
        _make_rich_program(TaskType.CLASSIFICATION),
        _make_constant_program(),
        _make_nonlinear_program(),
    ],
)
def test_c_emitter_compile_dll_and_main(tmp_path: Path, program: DecompiledProgram) -> None:
    if find_c_compiler() is None:
        pytest.skip("No C99 compiler available on this host")

    c_src, h_src = emit_c(program)
    assert "#include <math.h>" in c_src
    assert "#include <stddef.h>" in c_src
    assert "#ifdef GLASSBOX_MAIN" in c_src
    assert "GLASSBOX_API int glassbox_infer(" in h_src
    assert "GLASSBOX_API GlassBoxOutput glassbox_infer_struct(" in h_src
    assert "GLASSBOX_API double glassbox_bench_loop(" in h_src

    # Compile standalone GLASSBOX_MAIN binary with strict zero-warning flags
    exe_path, cc_name = compile_c_executable(
        c_src,
        tmp_path / "model_main",
        h_source=h_src,
        strict_warnings=True,
    )
    assert cc_name in {"gcc", "clang", "ziglang cc"}
    main_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert "PASS:" in main_run.stdout

    # Compile shared library (.dll / .so) with strict zero-warning flags
    dll_path, _ = compile_c_shared(
        c_src,
        tmp_path / "model",
        h_source=h_src,
        strict_warnings=True,
    )

    feat_names = sanitize_identifiers(program.domain.feature_names, prefix="x")
    out_names = sanitize_identifiers(program.domain.output_names, prefix="y")

    class CInput(ctypes.Structure):
        _fields_ = [(name, ctypes.c_double) for name in feat_names]

    class COutput(ctypes.Structure):
        _fields_ = [
            *[(name, ctypes.c_double) for name in out_names],
            ("regime_id", ctypes.c_int),
            ("class_id", ctypes.c_int),
        ]

    lib = ctypes.CDLL(str(dll_path))
    try:
        lib.glassbox_infer.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.glassbox_infer.restype = ctypes.c_int
        lib.glassbox_infer_struct.argtypes = [ctypes.POINTER(CInput)]
        lib.glassbox_infer_struct.restype = COutput
        lib.glassbox_bench_loop.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.glassbox_bench_loop.restype = ctypes.c_double

        rng = np.random.default_rng(7)
        lows = np.array(program.domain.lower_bounds, dtype=np.float64)
        highs = np.array(program.domain.upper_bounds, dtype=np.float64)
        X = rng.uniform(lows, highs, size=(32, program.domain.input_dim))
        ref_preds, ref_regimes = program.predict_with_regimes(X)
        ref_classes = (
            program.predict_classes(X)
            if program.task_type == TaskType.CLASSIFICATION
            else None
        )

        in_dim = program.domain.input_dim
        out_dim = program.domain.output_dim
        for idx, row in enumerate(X):
            x_buf = (ctypes.c_double * in_dim)(*[float(v) for v in row])
            y_buf = (ctypes.c_double * out_dim)()
            reg = ctypes.c_int(-1)
            rc = lib.glassbox_infer(x_buf, y_buf, ctypes.byref(reg))
            assert rc == 0
            assert reg.value == int(ref_regimes[idx])
            c_vals = [float(y_buf[j]) for j in range(out_dim)]
            assert np.max(np.abs(np.array(c_vals) - ref_preds[idx])) < 1e-9

            s_in = CInput(*[float(v) for v in row])
            s_out = lib.glassbox_infer_struct(ctypes.byref(s_in))
            assert s_out.regime_id == int(ref_regimes[idx])
            s_vals = [float(getattr(s_out, oname)) for oname in out_names]
            assert np.max(np.abs(np.array(s_vals) - ref_preds[idx])) < 1e-9
            if ref_classes is not None:
                assert s_out.class_id == int(ref_classes[idx])
            else:
                assert s_out.class_id == -1

        # Verify pure-C batch loop helper `glassbox_bench_loop`
        flat_x = [float(v) for row in X for v in row]
        xs_buf = (ctypes.c_double * len(flat_x))(*flat_x)
        last_buf = (ctypes.c_double * out_dim)()
        acc = float(lib.glassbox_bench_loop(xs_buf, len(X), 4, last_buf))
        assert math.isfinite(acc)
        last_vals = [float(last_buf[j]) for j in range(out_dim)]
        assert np.max(np.abs(np.array(last_vals) - ref_preds[-1])) < 1e-9
        assert float(lib.glassbox_bench_loop(xs_buf, 0, 4, last_buf)) == 0.0
    finally:
        if sys.platform == "win32":
            try:
                _ctypes.FreeLibrary(lib._handle)
            except OSError:
                pass


@pytest.mark.parametrize(
    "program",
    [
        _make_rich_program(TaskType.REGRESSION),
        _make_rich_program(TaskType.CLASSIFICATION),
        _make_constant_program(),
        _make_nonlinear_program(),
    ],
)
def test_python_emitter_exec_and_main(tmp_path: Path, program: DecompiledProgram) -> None:
    py_src = emit_python(program)

    # Verify zero third-party imports
    tree = ast.parse(py_src)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imports == {"math"}
    assert from_imports <= {"__future__", "dataclasses"}

    ns: dict[str, object] = {}
    exec(compile(py_src, "<emitted_py>", "exec"), ns)  # noqa: S102

    infer_fn = ns["infer"]
    infer_with_regime_fn = ns["infer_with_regime"]
    infer_struct_fn = ns["infer_struct"]

    rng = np.random.default_rng(11)
    lows = np.array(program.domain.lower_bounds, dtype=np.float64)
    highs = np.array(program.domain.upper_bounds, dtype=np.float64)
    X = rng.uniform(lows, highs, size=(32, program.domain.input_dim))
    ref_preds, ref_regimes = program.predict_with_regimes(X)
    ref_classes = (
        program.predict_classes(X)
        if program.task_type == TaskType.CLASSIFICATION
        else None
    )

    for idx, row in enumerate(X):
        row_list = [float(v) for v in row]
        vals = infer_fn(row_list)
        vals2, rid, cid = infer_with_regime_fn(row_list)
        struct_out = infer_struct_fn(row_list)
        assert vals == vals2
        assert rid == int(ref_regimes[idx])
        assert np.max(np.abs(np.array(vals) - ref_preds[idx])) < 1e-9
        assert struct_out.to_list() == vals
        assert struct_out.regime_id == rid
        assert struct_out.class_id == cid
        if ref_classes is not None:
            assert cid == int(ref_classes[idx])
        else:
            assert cid is None

    py_path = tmp_path / "emitted_prog.py"
    py_path.write_text(py_src, encoding="utf-8")
    run = subprocess.run([sys.executable, str(py_path)], capture_output=True, text=True, check=True)
    assert "PASS:" in run.stdout


def test_rust_emitter_structure_and_classification() -> None:
    reg_prog = _make_rich_program(TaskType.REGRESSION)
    cls_prog = _make_rich_program(TaskType.CLASSIFICATION)

    rs_reg = emit_rust(reg_prog)
    rs_cls = emit_rust(cls_prog)

    for code in (rs_reg, rs_cls):
        assert code.splitlines()[1] == "#![cfg_attr(not(test), no_std)]"
        assert "#[derive(Debug, Clone, Copy)]" in code
        assert "pub struct Input {" in code
        assert "pub struct Output {" in code
        assert "pub fn infer(input: &Input) -> Output {" in code
        assert "#[cfg(test)]" in code
        assert "mod tests {" in code
        assert code.count("{") == code.count("}")
        assert code.count("(") == code.count(")")

    assert "pub enum ClassId" not in rs_reg
    assert "pub enum ClassId {" in rs_cls
    assert "Stay = 0," in rs_cls
    assert "Boost = 1," in rs_cls


def test_ir_emitter_json_and_mermaid() -> None:
    prog = _make_rich_program(TaskType.CLASSIFICATION)
    ir_json = emit_ir_json(prog)
    restored = DecompiledProgram.model_validate_json(ir_json)
    assert restored.sha256() == prog.sha256()

    mermaid = emit_mermaid(prog)
    lines = [line.strip() for line in mermaid.splitlines() if line.strip()]
    assert lines[0] == "flowchart TD"
    # No HTML tags allowed
    assert not re.search(r"</?(br|b|i|div|span|p)\b", mermaid, flags=re.IGNORECASE)
    # Every node declaration uses strictly double-quoted labels
    node_lines = [line for line in lines[1:] if "-->" not in line]
    assert len(node_lines) == 5  # 2 guards + 3 leaves
    for line in node_lines:
        assert re.match(r'^[nl]\d+\["[^"]+"\]$', line), f"Invalid Mermaid node: {line}"

    # Verify quotes and HTML tags in leaf names are sanitized safely
    nl_prog = _make_nonlinear_program()
    nl_mermaid = emit_mermaid(nl_prog)
    assert "<br>" not in nl_mermaid
    assert 'l0["' in nl_mermaid


def test_emit_all_keys_and_deterministic_golden_vectors() -> None:
    prog = _make_rich_program(TaskType.REGRESSION)
    gv1 = generate_golden_vectors(prog)
    gv2 = generate_golden_vectors(prog)
    assert len(gv1) == 5
    assert gv1 == gv2

    bundle1 = emit_all(prog)
    bundle2 = emit_all(prog)
    assert set(bundle1.keys()) == {"c", "h", "rust", "python", "mermaid", "ir_json"}
    assert bundle1 == bundle2
