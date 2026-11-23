"""Runtime latency and numerical fidelity benchmarks across execution targets."""

from __future__ import annotations

import ctypes
import io
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from glassbox.ir import DecompiledProgram
from glassbox.types import BenchmarkEntry

if TYPE_CHECKING:
    from glassbox.loader import LoadedModel

if sys.platform == "win32":
    import _ctypes


def _summarize_latencies(latencies_us: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(latencies_us, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    throughput = 1_000_000.0 / max(p50, 1e-6)
    return p50, p95, p99, throughput


def _time_single_calls(
    fn: Callable[[int], Any],
    n_samples: int,
    n_warmup: int,
    n_iters: int,
) -> list[float]:
    for i in range(max(0, n_warmup)):
        fn(i % n_samples)
    latencies_us: list[float] = []
    iters = max(1, n_iters)
    for i in range(iters):
        idx = i % n_samples
        t0 = time.perf_counter_ns()
        fn(idx)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)
    return latencies_us


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


def _benchmark_onnx(
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
    artifact_bytes = model_path.stat().st_size if model_path.exists() else model.metadata.size_bytes

    latencies = _time_single_calls(
        lambda idx: sess.run(None, {input_name: samples_f32[idx]}),
        len(samples_f32),
        n_warmup,
        n_iters,
    )
    p50, p95, p99, throughput = _summarize_latencies(latencies)
    return BenchmarkEntry(
        runtime="ONNX Runtime (CPU)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=throughput,
        artifact_bytes=int(artifact_bytes),
        speedup_vs_onnx=1.0,
        max_diff_vs_ref=max_diff,
        available=True,
        note="batch=1 vs model.predict reference",
    )


def _benchmark_pytorch(
    model: LoadedModel,
    X_test: np.ndarray,
    ref_nn: np.ndarray,
    onnx_p50: float,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    seq, artifact_bytes = _build_torch_sequential(model)
    if seq is not None:
        samples_t = [torch.from_numpy(X_test[i : i + 1].astype(np.float32)) for i in range(len(X_test))]
        with torch.inference_mode():
            Y_pt = np.vstack([seq(s).numpy() for s in samples_t]).astype(np.float64)
            max_diff = float(np.max(np.abs(Y_pt - ref_nn)))
            latencies = _time_single_calls(
                lambda idx: seq(samples_t[idx]),
                len(samples_t),
                n_warmup,
                n_iters,
            )
        note = "reconstructed nn.Sequential vs model.predict"
    else:
        samples_f64 = [X_test[i : i + 1] for i in range(len(X_test))]
        Y_pt = np.vstack([model.predict(s) for s in samples_f64])
        max_diff = float(np.max(np.abs(Y_pt - ref_nn)))
        latencies = _time_single_calls(
            lambda idx: model.predict(samples_f64[idx]),
            len(samples_f64),
            n_warmup,
            n_iters,
        )
        note = "fallback to model.predict (non-MLP)"

    p50, p95, p99, throughput = _summarize_latencies(latencies)
    return BenchmarkEntry(
        runtime="PyTorch (Eager CPU)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=throughput,
        artifact_bytes=int(artifact_bytes),
        speedup_vs_onnx=onnx_p50 / max(p50, 1e-6),
        max_diff_vs_ref=max_diff,
        available=True,
        note=note,
    )


def _benchmark_python(
    emitted_code: dict[str, str],
    X_test: np.ndarray,
    ref_prog: np.ndarray,
    onnx_p50: float,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    py_src = emitted_code.get("python", "")
    if not py_src.strip():
        return BenchmarkEntry(
            runtime="Decompiled Python (Zero-Dep)",
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_ips=0.0,
            artifact_bytes=0,
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
            artifact_bytes=len(py_src.encode("utf-8")),
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

    latencies = _time_single_calls(
        lambda idx: infer_fn(samples_py[idx]),
        len(samples_py),
        n_warmup,
        n_iters,
    )
    p50, p95, p99, throughput = _summarize_latencies(latencies)
    return BenchmarkEntry(
        runtime="Decompiled Python (Zero-Dep)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=throughput,
        artifact_bytes=len(py_src.encode("utf-8")),
        speedup_vs_onnx=onnx_p50 / max(p50, 1e-6),
        max_diff_vs_ref=max_diff,
        available=True,
        note="zero-dep Python vs program.predict",
    )


def _benchmark_c99(
    program: DecompiledProgram,
    emitted_code: dict[str, str],
    X_test: np.ndarray,
    ref_prog: np.ndarray,
    onnx_p50: float,
    work_dir: Path | None,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkEntry:
    c_src = emitted_code.get("c", "")
    h_src = emitted_code.get("h", "")
    c_bytes = len(c_src.encode("utf-8"))
    if not c_src.strip():
        return BenchmarkEntry(
            runtime="Decompiled C99 (Native DLL)",
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_ips=0.0,
            artifact_bytes=0,
            speedup_vs_onnx=0.0,
            max_diff_vs_ref=0.0,
            available=False,
            note="missing emitted C99 code",
        )

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    ext = ".dll" if sys.platform == "win32" else ".so"
    with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "glassbox_model.c"
        dll_path = tmp_path / f"glassbox_model{ext}"
        c_path.write_text(c_src, encoding="utf-8")

        if h_src.strip():
            # Ensure any local `#include "header.h"` directive resolves in the build dir.
            inc_match = re.search(r'#include\s+"([^"]+)"', c_src)
            if inc_match:
                (tmp_path / Path(inc_match.group(1)).name).write_text(h_src, encoding="utf-8")
            (tmp_path / "glassbox_model.h").write_text(h_src, encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "ziglang",
            "cc",
            "-O3",
            "-shared",
            "-std=c99",
            str(c_path),
            "-o",
            str(dll_path),
            "-lm",
        ]
        if sys.platform != "win32":
            cmd.append("-fPIC")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return BenchmarkEntry(
                runtime="Decompiled C99 (Native DLL)",
                p50_us=0.0,
                p95_us=0.0,
                p99_us=0.0,
                throughput_ips=0.0,
                artifact_bytes=c_bytes,
                speedup_vs_onnx=0.0,
                max_diff_vs_ref=0.0,
                available=False,
                note=f"C compiler unavailable: {exc}",
            )

        if proc.returncode != 0 or not dll_path.exists():
            err_msg = (proc.stderr or proc.stdout).strip()[:200]
            return BenchmarkEntry(
                runtime="Decompiled C99 (Native DLL)",
                p50_us=0.0,
                p95_us=0.0,
                p99_us=0.0,
                throughput_ips=0.0,
                artifact_bytes=c_bytes,
                speedup_vs_onnx=0.0,
                max_diff_vs_ref=0.0,
                available=False,
                note=f"C99 compile failed: {err_msg}",
            )

        dll_bytes = dll_path.stat().st_size
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
            c_inputs = [(ctypes.c_double * in_dim)(*row.tolist()) for row in X_test]
            out_buf = (ctypes.c_double * out_dim)()
            reg_buf = ctypes.c_int()

            y_rows: list[list[float]] = []
            for x_buf in c_inputs:
                infer_c(x_buf, out_buf, ctypes.byref(reg_buf))
                y_rows.append([out_buf[j] for j in range(out_dim)])
            Y_c = np.asarray(y_rows, dtype=np.float64)
            max_diff = float(np.max(np.abs(Y_c - ref_prog)))

            reg_ptr = ctypes.byref(reg_buf)
            latencies = _time_single_calls(
                lambda idx: infer_c(c_inputs[idx], out_buf, reg_ptr),
                len(c_inputs),
                n_warmup,
                n_iters,
            )
        finally:
            if sys.platform == "win32":
                try:
                    _ctypes.FreeLibrary(lib._handle)
                except OSError:
                    pass

    p50, p95, p99, throughput = _summarize_latencies(latencies)
    return BenchmarkEntry(
        runtime="Decompiled C99 (Native DLL)",
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        throughput_ips=throughput,
        artifact_bytes=int(dll_bytes),
        speedup_vs_onnx=onnx_p50 / max(p50, 1e-6),
        max_diff_vs_ref=max_diff,
        available=True,
        note="ziglang cc -O3 -std=c99 ctypes single-call vs program.predict",
    )


def run_benchmarks(
    model: LoadedModel,
    program: DecompiledProgram,
    emitted_code: dict[str, str],
    work_dir: Path | None = None,
    n_warmup: int = 100,
    n_iters: int = 2000,
) -> list[BenchmarkEntry]:
    """Benchmark ONNX Runtime, PyTorch, decompiled Python, and native C99 DLL."""
    rng = np.random.default_rng(12345)
    low = np.asarray(program.domain.lower_bounds, dtype=np.float64)
    high = np.asarray(program.domain.upper_bounds, dtype=np.float64)
    X_test = rng.uniform(low, high, size=(200, program.domain.input_dim))

    ref_nn = model.predict(X_test)
    ref_prog = program.predict(X_test)

    onnx_entry = _benchmark_onnx(model, X_test, ref_nn, n_warmup, n_iters)
    onnx_p50 = max(onnx_entry.p50_us, 1e-6)

    pytorch_entry = _benchmark_pytorch(model, X_test, ref_nn, onnx_p50, n_warmup, n_iters)
    python_entry = _benchmark_python(emitted_code, X_test, ref_prog, onnx_p50, n_warmup, n_iters)
    c99_entry = _benchmark_c99(
        program,
        emitted_code,
        X_test,
        ref_prog,
        onnx_p50,
        work_dir,
        n_warmup,
        n_iters,
    )
    return [onnx_entry, pytorch_entry, python_entry, c99_entry]


__all__ = ["run_benchmarks"]
