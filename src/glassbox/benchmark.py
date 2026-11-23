"""Runtime latency, sustained throughput, and numerical fidelity benchmarks."""

from __future__ import annotations

import ctypes
import io
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from glassbox.emitters import compile_c_shared, find_c_compiler
from glassbox.ir import DecompiledProgram
from glassbox.types import BenchmarkEntry

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel

if sys.platform == "win32":
    import _ctypes


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def _measure_calls(
    fn: Callable[[int], Any],
    n_slots: int,
    n_warmup: int,
    n_iters: int,
    batch_size: int = 1,
) -> tuple[float, float, float, float]:
    """Return (p50_us_per_sample, p95_us_per_sample, p99_us_per_sample, measured_throughput_ips)."""
    for i in range(max(0, n_warmup)):
        fn(i % n_slots)

    iters = max(1, n_iters)
    latencies_us: list[float] = []
    scale = 1000.0 * max(1, batch_size)
    for i in range(iters):
        idx = i % n_slots
        t0 = time.perf_counter_ns()
        fn(idx)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / scale)

    p50, p95, p99 = _percentiles(latencies_us)

    # Measure uninterrupted sustained wall-clock throughput without per-call timer overhead.
    t_start = time.perf_counter_ns()
    for i in range(iters):
        fn(i % n_slots)
    t_end = time.perf_counter_ns()
    wall_s = max((t_end - t_start) / 1e9, 1e-9)
    measured_ips = float((iters * max(1, batch_size)) / wall_s)
    return p50, p95, p99, measured_ips


def _build_torch_sequential(model: LoadedModel) -> tuple[nn.Sequential | None, int]:
    layers = model.extract_mlp_layers()
    if not layers:
        return None, int(model.metadata.total_params * 4)

    modules: list[nn.Module] = []
    for layer in layers:
        in_f = int(layer.weight.shape[1])
        out_f = int(layer.weight.shape[0])
        lin = nn.Linear(in_f, out_f)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(layer.weight, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(layer.bias, dtype=torch.float32))
        modules.append(lin)
        if layer.activation == "Relu":
            modules.append(nn.ReLU())
        elif layer.activation == "Tanh":
            modules.append(nn.Tanh())
        elif layer.activation == "Sigmoid":
            modules.append(nn.Sigmoid())

    seq = nn.Sequential(*modules)
    seq.eval()
    buf = io.BytesIO()
    torch.save(seq.state_dict(), buf)
    return seq, len(buf.getvalue())


def _benchmark_onnx_batch1(
    model: LoadedModel,
    X_test: np.ndarray,
    ref_nn: np.ndarray,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    sess = model.session
    input_name = sess.get_inputs()[0].name
    samples_f32 = [X_test[i : i + 1].astype(np.float32) for i in range(len(X_test))]
    Y_onnx = np.vstack([sess.run(None, {input_name: s})[0] for s in samples_f32]).astype(
        np.float64
    )
    max_diff = float(np.max(np.abs(Y_onnx - ref_nn)))

    model_path = Path(model.metadata.path)
    artifact_bytes = int(
        model_path.stat().st_size if model_path.exists() else model.metadata.size_bytes
    )

    p50, p95, p99, measured_ips = _measure_calls(
        lambda idx: sess.run(None, {input_name: samples_f32[idx]}),
        len(samples_f32),
        n_warmup,
        n_iters,
        batch_size=1,
    )
    return BenchmarkEntry(
        runtime="ONNX Runtime (CPU, batch=1)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=measured_ips,
        measured_throughput_ips=measured_ips,
        batch_size=1,
        artifact_bytes=artifact_bytes,
        binary_bytes=artifact_bytes,
        speedup_vs_onnx=1.0,
        max_diff_vs_ref=max_diff,
        available=True,
        note="single-sample control-loop latency vs model.predict",
    )


def _benchmark_onnx_batch64(
    model: LoadedModel,
    X_test: np.ndarray,
    ref_nn: np.ndarray,
    onnx_b1_p50: float,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    sess = model.session
    input_name = sess.get_inputs()[0].name
    X_f32 = X_test.astype(np.float32)
    Y_onnx_full = np.asarray(sess.run(None, {input_name: X_f32})[0], dtype=np.float64)
    max_diff = float(np.max(np.abs(Y_onnx_full - ref_nn)))

    batch_size = 64
    n_test = len(X_f32)
    batches_f32 = [
        np.take(X_f32, np.arange(k * batch_size, (k + 1) * batch_size) % n_test, axis=0)
        for k in range(16)
    ]

    model_path = Path(model.metadata.path)
    artifact_bytes = int(
        model_path.stat().st_size if model_path.exists() else model.metadata.size_bytes
    )
    batch_warmup = max(5, n_warmup // 4)
    batch_iters = max(25, n_iters // 8)

    p50, p95, p99, measured_ips = _measure_calls(
        lambda idx: sess.run(None, {input_name: batches_f32[idx]}),
        len(batches_f32),
        batch_warmup,
        batch_iters,
        batch_size=batch_size,
    )
    return BenchmarkEntry(
        runtime="ONNX Runtime (CPU, batch=64)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=measured_ips,
        measured_throughput_ips=measured_ips,
        batch_size=batch_size,
        artifact_bytes=artifact_bytes,
        binary_bytes=artifact_bytes,
        speedup_vs_onnx=onnx_b1_p50 / max(p50, 1e-6),
        max_diff_vs_ref=max_diff,
        available=True,
        note="amortized per-sample latency across batch=64",
    )


def _benchmark_pytorch(
    model: LoadedModel,
    X_test: np.ndarray,
    ref_nn: np.ndarray,
    onnx_b1_p50: float,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    seq, artifact_bytes = _build_torch_sequential(model)
    if seq is not None:
        samples_t = [
            torch.from_numpy(X_test[i : i + 1].astype(np.float32)) for i in range(len(X_test))
        ]
        with torch.inference_mode():
            Y_pt = np.vstack([seq(s).numpy() for s in samples_t]).astype(np.float64)
            max_diff = float(np.max(np.abs(Y_pt - ref_nn)))
            p50, p95, p99, measured_ips = _measure_calls(
                lambda idx: seq(samples_t[idx]),
                len(samples_t),
                n_warmup,
                n_iters,
                batch_size=1,
            )
        note = "reconstructed nn.Sequential eager batch=1 vs model.predict"
    else:
        samples_f64 = [X_test[i : i + 1] for i in range(len(X_test))]
        Y_pt = np.vstack([model.predict(s) for s in samples_f64])
        max_diff = float(np.max(np.abs(Y_pt - ref_nn)))
        p50, p95, p99, measured_ips = _measure_calls(
            lambda idx: model.predict(samples_f64[idx]),
            len(samples_f64),
            n_warmup,
            n_iters,
            batch_size=1,
        )
        note = "fallback to model.predict (non-MLP)"

    return BenchmarkEntry(
        runtime="PyTorch (Eager CPU, batch=1)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=measured_ips,
        measured_throughput_ips=measured_ips,
        batch_size=1,
        artifact_bytes=int(artifact_bytes),
        binary_bytes=int(artifact_bytes),
        speedup_vs_onnx=onnx_b1_p50 / max(p50, 1e-6),
        max_diff_vs_ref=max_diff,
        available=True,
        note=note,
    )


def _benchmark_python(
    emitted_code: dict[str, str],
    X_test: np.ndarray,
    ref_prog: np.ndarray,
    onnx_b1_p50: float,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    py_src = emitted_code.get("python", "")
    py_bytes = len(py_src.encode("utf-8"))
    if not py_src.strip():
        return BenchmarkEntry(
            runtime="Decompiled Python (Zero-Dep)",
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_ips=0.0,
            measured_throughput_ips=0.0,
            batch_size=1,
            artifact_bytes=0,
            source_bytes=0,
            binary_bytes=0,
            speedup_vs_onnx=0.0,
            max_diff_vs_ref=0.0,
            available=False,
            note="missing emitted python code",
        )

    ns: dict[str, Any] = {}
    exec(compile(py_src, "<glassbox_emitted_py>", "exec"), ns)  # noqa: S102
    infer_fn = ns.get("infer")
    if not callable(infer_fn):
        return BenchmarkEntry(
            runtime="Decompiled Python (Zero-Dep)",
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_ips=0.0,
            measured_throughput_ips=0.0,
            batch_size=1,
            artifact_bytes=py_bytes,
            source_bytes=py_bytes,
            binary_bytes=0,
            speedup_vs_onnx=0.0,
            max_diff_vs_ref=0.0,
            available=False,
            note="emitted python lacks callable infer(x)",
        )

    samples_py = [row.tolist() for row in X_test]
    probe = infer_fn(samples_py[0])
    unwrap_tuple = isinstance(probe, tuple)

    def _eval_one(row: list[float]) -> list[float]:
        out = infer_fn(row)
        return list(out[0]) if unwrap_tuple else list(out)

    Y_py = np.asarray([_eval_one(r) for r in samples_py], dtype=np.float64)
    max_diff = float(np.max(np.abs(Y_py - ref_prog)))

    p50, p95, p99, measured_ips = _measure_calls(
        lambda idx: infer_fn(samples_py[idx]),
        len(samples_py),
        n_warmup,
        n_iters,
        batch_size=1,
    )
    return BenchmarkEntry(
        runtime="Decompiled Python (Zero-Dep)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=measured_ips,
        measured_throughput_ips=measured_ips,
        batch_size=1,
        artifact_bytes=py_bytes,
        source_bytes=py_bytes,
        binary_bytes=0,
        speedup_vs_onnx=onnx_b1_p50 / max(p50, 1e-6),
        max_diff_vs_ref=max_diff,
        available=True,
        note="zero-dep Python vs program.predict",
    )


def _unavailable_c_entries(c_bytes: int, note: str, compiler: str = "") -> list[BenchmarkEntry]:
    return [
        BenchmarkEntry(
            runtime="Decompiled C99 (ctypes FFI)",
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_ips=0.0,
            measured_throughput_ips=0.0,
            batch_size=1,
            artifact_bytes=c_bytes,
            source_bytes=c_bytes,
            binary_bytes=0,
            compiler=compiler,
            speedup_vs_onnx=0.0,
            max_diff_vs_ref=0.0,
            available=False,
            note=note,
        ),
        BenchmarkEntry(
            runtime="Decompiled C99 (Native C Loop)",
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_ips=0.0,
            measured_throughput_ips=0.0,
            batch_size=1,
            native_loop_p50_ns=None,
            artifact_bytes=c_bytes,
            source_bytes=c_bytes,
            binary_bytes=0,
            compiler=compiler,
            speedup_vs_onnx=0.0,
            max_diff_vs_ref=0.0,
            available=False,
            note=note,
        ),
    ]


def _benchmark_c99_targets(
    program: DecompiledProgram,
    emitted_code: dict[str, str],
    X_test: np.ndarray,
    ref_prog: np.ndarray,
    onnx_b1_p50: float,
    work_dir: Path | None,
    n_warmup: int,
    n_iters: int,
) -> list[BenchmarkEntry]:
    c_src = emitted_code.get("c", "")
    h_src = emitted_code.get("h", "")
    c_bytes = len(c_src.encode("utf-8"))
    if not c_src.strip():
        return _unavailable_c_entries(0, "missing emitted C99 code")

    compiler_info = find_c_compiler()
    if compiler_info is None:
        return _unavailable_c_entries(c_bytes, "no C99 compiler found (gcc/clang/ziglang)")

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        try:
            dll_path, cc_name = compile_c_shared(
                c_src,
                tmp_path / "glassbox_model",
                h_source=h_src,
                strict_warnings=True,
            )
        except (RuntimeError, OSError) as exc:
            return _unavailable_c_entries(c_bytes, f"C99 compile failed: {exc}", compiler_info[1])

        dll_bytes = int(dll_path.stat().st_size)
        lib = ctypes.CDLL(str(dll_path))
        try:
            infer_c = lib.glassbox_infer
            infer_c.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
            ]
            infer_c.restype = ctypes.c_int

            in_dim = program.domain.input_dim
            out_dim = program.domain.output_dim
            n_samples = len(X_test)
            c_inputs = [(ctypes.c_double * in_dim)(*row.tolist()) for row in X_test]
            out_buf = (ctypes.c_double * out_dim)()
            reg_buf = ctypes.c_int()

            y_rows: list[list[float]] = []
            for x_buf in c_inputs:
                infer_c(x_buf, out_buf, ctypes.byref(reg_buf))
                y_rows.append([out_buf[j] for j in range(out_dim)])
            Y_c = np.asarray(y_rows, dtype=np.float64)
            max_diff_ffi = float(np.max(np.abs(Y_c - ref_prog)))

            reg_ptr = ctypes.byref(reg_buf)
            ffi_p50, ffi_p95, ffi_p99, ffi_ips = _measure_calls(
                lambda idx: infer_c(c_inputs[idx], out_buf, reg_ptr),
                len(c_inputs),
                n_warmup,
                n_iters,
                batch_size=1,
            )

            ffi_entry = BenchmarkEntry(
                runtime="Decompiled C99 (ctypes FFI)",
                p50_us=ffi_p50,
                p95_us=ffi_p95,
                p99_us=ffi_p99,
                throughput_ips=ffi_ips,
                measured_throughput_ips=ffi_ips,
                batch_size=1,
                artifact_bytes=dll_bytes,
                source_bytes=c_bytes,
                binary_bytes=dll_bytes,
                compiler=cc_name,
                speedup_vs_onnx=onnx_b1_p50 / max(ffi_p50, 1e-6),
                max_diff_vs_ref=max_diff_ffi,
                available=True,
                note=f"{cc_name} -O3 -std=c99 single-call via Python ctypes FFI",
            )

            bench_loop = getattr(lib, "glassbox_bench_loop", None)
            if bench_loop is None:
                native_entry = BenchmarkEntry(
                    runtime="Decompiled C99 (Native C Loop)",
                    p50_us=0.0,
                    p95_us=0.0,
                    p99_us=0.0,
                    throughput_ips=0.0,
                    measured_throughput_ips=0.0,
                    batch_size=1,
                    native_loop_p50_ns=None,
                    artifact_bytes=dll_bytes,
                    source_bytes=c_bytes,
                    binary_bytes=dll_bytes,
                    compiler=cc_name,
                    speedup_vs_onnx=0.0,
                    max_diff_vs_ref=max_diff_ffi,
                    available=False,
                    note="glassbox_bench_loop symbol missing",
                )
            else:
                bench_loop.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_double),
                ]
                bench_loop.restype = ctypes.c_double

                flat_vals = X_test.reshape(-1).tolist()
                xs_flat = (ctypes.c_double * len(flat_vals))(*flat_vals)
                out_last = (ctypes.c_double * out_dim)()
                n_repeats = max(50, (max(200, n_iters) * 5) // n_samples)
                calls_per_run = n_samples * n_repeats

                for _ in range(3):
                    bench_loop(xs_flat, n_samples, n_repeats, out_last)

                last_vals = np.asarray([out_last[j] for j in range(out_dim)], dtype=np.float64)
                last_diff = float(np.max(np.abs(last_vals - ref_prog[-1])))
                max_diff_native = max(max_diff_ffi, last_diff)

                ns_per_call_runs: list[float] = []
                total_ns = 0
                total_calls = 0
                for _ in range(25):
                    t0 = time.perf_counter_ns()
                    bench_loop(xs_flat, n_samples, n_repeats, out_last)
                    t1 = time.perf_counter_ns()
                    elapsed_ns = max(1, t1 - t0)
                    ns_per_call_runs.append(elapsed_ns / calls_per_run)
                    total_ns += elapsed_ns
                    total_calls += calls_per_run

                p50_ns, p95_ns, p99_ns = _percentiles(ns_per_call_runs)
                p50_us = p50_ns / 1000.0
                p95_us = p95_ns / 1000.0
                p99_us = p99_ns / 1000.0
                native_ips = float(total_calls / max(total_ns / 1e9, 1e-9))

                native_entry = BenchmarkEntry(
                    runtime="Decompiled C99 (Native C Loop)",
                    p50_us=p50_us,
                    p95_us=p95_us,
                    p99_us=p99_us,
                    throughput_ips=native_ips,
                    measured_throughput_ips=native_ips,
                    batch_size=1,
                    native_loop_p50_ns=p50_ns,
                    artifact_bytes=dll_bytes,
                    source_bytes=c_bytes,
                    binary_bytes=dll_bytes,
                    compiler=cc_name,
                    speedup_vs_onnx=onnx_b1_p50 / max(p50_us, 1e-6),
                    max_diff_vs_ref=max_diff_native,
                    available=True,
                    note=f"{cc_name} -O3 -std=c99 in-C tight loop ({p50_ns:.1f} ns/call, 0 FFI)",
                )
        finally:
            if sys.platform == "win32":
                try:
                    _ctypes.FreeLibrary(lib._handle)
                except OSError:
                    pass

    return [ffi_entry, native_entry]


def run_benchmarks(
    model: LoadedModel,
    program: DecompiledProgram,
    emitted_code: dict[str, str],
    work_dir: Path | None = None,
    n_warmup: int = 100,
    n_iters: int = 2000,
) -> list[BenchmarkEntry]:
    """Benchmark ONNX (batch=1, batch=64), PyTorch, Python, C99 ctypes FFI, and C99 native loop."""
    rng = np.random.default_rng(12345)
    low = np.asarray(program.domain.lower_bounds, dtype=np.float64)
    high = np.asarray(program.domain.upper_bounds, dtype=np.float64)
    X_test = rng.uniform(low, high, size=(200, program.domain.input_dim))

    ref_nn = model.predict(X_test)
    ref_prog = program.predict(X_test)

    onnx_b1 = _benchmark_onnx_batch1(model, X_test, ref_nn, n_warmup, n_iters)
    onnx_b1_p50 = max(onnx_b1.p50_us, 1e-6)

    onnx_b64 = _benchmark_onnx_batch64(model, X_test, ref_nn, onnx_b1_p50, n_warmup, n_iters)
    pytorch_b1 = _benchmark_pytorch(model, X_test, ref_nn, onnx_b1_p50, n_warmup, n_iters)
    python_entry = _benchmark_python(
        emitted_code, X_test, ref_prog, onnx_b1_p50, n_warmup, n_iters
    )
    c99_entries = _benchmark_c99_targets(
        program,
        emitted_code,
        X_test,
        ref_prog,
        onnx_b1_p50,
        work_dir,
        n_warmup,
        n_iters,
    )
    return [onnx_b1, onnx_b64, pytorch_b1, python_entry, *c99_entries]


__all__ = ["run_benchmarks"]
