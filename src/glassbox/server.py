"""FastAPI backend for the GlassBox Interactive Decompiler Studio."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
import tempfile
import threading
import uuid
from pathlib import Path
from queue import Queue
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, FiniteFloat, ValidationError

from glassbox import __version__
from glassbox.cli import run_decompile_pipeline
from glassbox.ir import DecompileResult
from glassbox.types import (
    CEGISIteration,
    DecompileConfig,
    DomainSpec,
    TaskType,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_LOG = logging.getLogger(__name__)


class DecompileRequest(BaseModel):
    """Request payload for synchronous or streaming decompilation."""

    model_id: str
    config: DecompileConfig | None = None
    run_benchmarks: bool = True
    run_pareto: bool = False


class SliceRequest(BaseModel):
    """Request payload for 2D input-space slice evaluation."""

    run_id: str
    dim_x: int
    dim_y: int
    fixed_values: list[FiniteFloat] | None = None
    resolution: int = 48


class ExplainRequest(BaseModel):
    """Request payload for deterministic single-input decision explanation."""

    run_id: str
    x: list[FiniteFloat]


def _z3_available() -> bool:
    return importlib.util.find_spec("z3") is not None


def _sanitize_id(raw: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(raw).stem).strip("_")
    return stem or f"model_{uuid.uuid4().hex[:6]}"


def _sse_frame(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _check_domain_point(domain: DomainSpec, point: list[float]) -> None:
    if len(point) != domain.input_dim:
        raise HTTPException(400, f"Expected {domain.input_dim} input coordinates")
    for name, value, lo, hi in zip(domain.feature_names, point, domain.lower_bounds, domain.upper_bounds):
        if not lo <= value <= hi:
            raise HTTPException(400, f"{name} must be within [{lo}, {hi}]")


def compute_slice_payload(
    model: Any,
    result: DecompileResult,
    *,
    dim_x: int,
    dim_y: int,
    fixed_values: list[float] | None = None,
    resolution: int = 48,
) -> dict[str, Any]:
    """Evaluate LoadedModel and DecompiledProgram on an N x N 2D input-space slice."""
    domain = result.metadata.domain
    d = domain.input_dim
    if not (0 <= dim_x < d) or not (0 <= dim_y < d):
        raise HTTPException(
            status_code=400,
            detail=f"dim_x ({dim_x}) and dim_y ({dim_y}) must be in [0, {d - 1}]",
        )

    if dim_x == dim_y:
        raise HTTPException(400, "A 2D slice requires two different dimensions")

    res = max(8, min(80, int(resolution)))
    if fixed_values is None:
        base = [
            0.5 * (lo + hi) for lo, hi in zip(domain.lower_bounds, domain.upper_bounds)
        ]
    else:
        _check_domain_point(domain, fixed_values)
        base = [float(v) for v in fixed_values]

    x_vals = np.linspace(domain.lower_bounds[dim_x], domain.upper_bounds[dim_x], res)
    y_vals = np.linspace(domain.lower_bounds[dim_y], domain.upper_bounds[dim_y], res)

    grid = np.tile(np.asarray(base, dtype=np.float64), (res * res, 1))
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid[:, dim_x] = xx.ravel()
    grid[:, dim_y] = yy.ravel()

    nn_out = model.predict(grid)
    prog_out, regimes = result.program.predict_with_regimes(grid)
    diff_flat = np.max(np.abs(nn_out - prog_out), axis=1)

    nn_grid = nn_out[:, 0].reshape(res, res).tolist()
    prog_grid = prog_out[:, 0].reshape(res, res).tolist()
    diff_grid = diff_flat.reshape(res, res).tolist()
    regime_grid = regimes.reshape(res, res).tolist()

    nn_class_grid: list[list[int]] | None = None
    prog_class_grid: list[list[int]] | None = None
    if domain.task_type == TaskType.CLASSIFICATION:
        nn_cls = model.predict_classes(grid)
        prog_cls = result.program.predict_classes(grid)
        nn_class_grid = nn_cls.reshape(res, res).tolist()
        prog_class_grid = prog_cls.reshape(res, res).tolist()

    projected = [
        {
            "iteration": cx.iteration,
            "source": cx.source,
            "x": float(cx.x[dim_x]),
            "y": float(cx.x[dim_y]),
            "max_abs_error": float(cx.max_abs_error),
            "regime_id": int(cx.regime_id),
            "repaired": bool(cx.repaired),
            "nn_output": [float(v) for v in cx.nn_output],
            "program_output": [float(v) for v in cx.program_output],
        }
        for cx in result.counterexamples
        if len(cx.x) == d
    ]

    return {
        "dim_x": dim_x,
        "dim_y": dim_y,
        "x_name": domain.feature_names[dim_x],
        "y_name": domain.feature_names[dim_y],
        "fixed_values": base,
        "resolution": res,
        "task_type": domain.task_type.value,
        "x_vals": [float(v) for v in x_vals],
        "y_vals": [float(v) for v in y_vals],
        "nn_grid": nn_grid,
        "prog_grid": prog_grid,
        "diff_grid": diff_grid,
        "regime_grid": regime_grid,
        "nn_class_grid": nn_class_grid,
        "prog_class_grid": prog_class_grid,
        "counterexamples_projected": projected,
    }


def create_app(
    models_dir: Path | None = None,
    studio_dir: Path | None = None,
    site_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the GlassBox Studio FastAPI application."""
    root = Path(__file__).resolve().parents[2]
    resolved_models = (models_dir or (root / "models")).resolve()
    resolved_studio = (studio_dir or (root / "studio")).resolve()
    resolved_site = (site_dir or (root / "site")).resolve()
    resolved_bundles = (root / "bundles").resolve()
    upload_dir = Path(tempfile.mkdtemp(prefix="glassbox_uploads_"))

    app = FastAPI(title="GlassBox Studio API", version=__version__)

    uploaded_paths: dict[str, Path] = {}
    uploaded_domains: dict[str, DomainSpec] = {}
    model_cache: dict[str, Any] = {}
    run_cache: dict[str, dict[str, Any]] = {}
    bundle_lock = threading.Lock()

    def _load_model_entry(model_id: str) -> Any:
        from glassbox.loader import load_onnx

        if not _SAFE_ID_RE.match(model_id):
            raise HTTPException(status_code=400, detail="Invalid model_id")
        if model_id in model_cache:
            return model_cache[model_id]
        if model_id in uploaded_paths:
            loaded = load_onnx(uploaded_paths[model_id], domain=uploaded_domains.get(model_id))
            model_cache[model_id] = loaded
            return loaded
        candidate = resolved_models / f"{model_id}.onnx"
        if candidate.is_file():
            loaded = load_onnx(candidate)
            model_cache[model_id] = loaded
            return loaded
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    def _load_meta_sidecar(onnx_path: Path) -> dict[str, Any]:
        meta_path = onnx_path.with_suffix(".meta.json")
        if not meta_path.is_file():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("Failed to parse sidecar %s: %s", meta_path, exc)
            return {}

    def _prebuilt_run_path(run_id: str) -> Path | None:
        if not _SAFE_ID_RE.match(run_id):
            return None
        candidate = resolved_studio / "demo-data" / "runs" / f"{run_id}.json"
        return candidate if candidate.is_file() else None

    def _load_run_entry(run_id: str) -> tuple[DecompileResult, Any]:
        if run_id in run_cache:
            entry = run_cache[run_id]
            return entry["result"], entry["model"]
        prebuilt = _prebuilt_run_path(run_id)
        if prebuilt is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        raw = json.loads(prebuilt.read_text(encoding="utf-8"))
        payload = raw["result"] if isinstance(raw, dict) and "result" in raw else raw
        result = DecompileResult.model_validate(payload)

        model_obj: Any = None
        model_id = result.metadata.name
        try:
            model_obj = _load_model_entry(model_id)
        except HTTPException:
            pass
        if model_obj is not None and model_obj.metadata.sha256 != result.certificate.model_sha256:
            model_obj = None
        if model_obj is not None:
            run_cache[run_id] = {"result": result, "model": model_obj}
        return result, model_obj

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "z3_available": _z3_available(),
        }

    @app.get("/api/models")
    def list_models() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _default_cfg(meta_dict: dict[str, Any], sidecar: dict[str, Any]) -> dict[str, Any]:
            raw_cfg = sidecar.get("recommended_config") or sidecar.get("default_config")
            if isinstance(raw_cfg, dict):
                return DecompileConfig.model_validate(raw_cfg).model_dump(mode="json")
            pwl = bool(meta_dict.get("is_piecewise_linear", True))
            task = meta_dict.get("domain", {}).get("task_type")
            return DecompileConfig(
                task_type=TaskType(task) if task else None,
                verify_smt=pwl,
                max_degree=1 if pwl else 2,
            ).model_dump(mode="json")

        if resolved_models.is_dir():
            for onnx_path in sorted(resolved_models.glob("*.onnx")):
                mid = onnx_path.stem
                loaded = _load_model_entry(mid)
                sidecar = _load_meta_sidecar(onnx_path)
                meta_dict = loaded.metadata.model_dump(mode="json")
                entries.append(
                    {
                        "id": mid,
                        "name": sidecar.get("name", loaded.metadata.name),
                        "description": sidecar.get("description", f"ONNX model {mid}"),
                        "metadata": meta_dict,
                        "default_config": _default_cfg(meta_dict, sidecar),
                        "has_prebuilt_run": _prebuilt_run_path(mid) is not None,
                    }
                )
                seen_ids.add(mid)

        for mid, path in uploaded_paths.items():
            if mid in seen_ids:
                continue
            loaded = _load_model_entry(mid)
            meta_dict = loaded.metadata.model_dump(mode="json")
            entries.append(
                {
                    "id": mid,
                    "name": loaded.metadata.name,
                    "description": f"Uploaded ONNX model ({path.name})",
                    "metadata": meta_dict,
                    "default_config": _default_cfg(meta_dict, {}),
                    "has_prebuilt_run": False,
                }
            )
            seen_ids.add(mid)

        demo_index = resolved_studio / "demo-data" / "index.json"
        if demo_index.is_file():
            try:
                index_data = json.loads(demo_index.read_text(encoding="utf-8"))
                for item in index_data.get("models", []):
                    mid = item.get("id")
                    if mid and mid not in seen_ids:
                        entries.append(item)
                        seen_ids.add(mid)
            except (OSError, json.JSONDecodeError) as exc:
                _LOG.warning("Failed to parse demo-data index %s: %s", demo_index, exc)

        return entries

    @app.post("/api/upload")
    async def upload_model(
        file: Annotated[UploadFile, File(...)],
        domain_json: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        filename = file.filename or ""
        if not filename.lower().endswith(".onnx"):
            raise HTTPException(status_code=400, detail="Only .onnx files are accepted")

        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="Uploaded file exceeds 20 MB limit")

        domain_spec: DomainSpec | None = None
        if domain_json:
            try:
                domain_spec = DomainSpec.model_validate_json(domain_json)
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid domain_json: {exc}") from exc

        base_id = _sanitize_id(filename)
        model_id = (
            base_id
            if base_id not in uploaded_paths and not (resolved_models / f"{base_id}.onnx").is_file()
            else f"{base_id}_{uuid.uuid4().hex[:6]}"
        )
        dest_path = upload_dir / f"{model_id}.onnx"

        try:
            import onnx

            proto = onnx.load_from_string(raw)
            onnx.checker.check_model(proto)
            dest_path.write_bytes(raw)
            from glassbox.loader import load_onnx

            loaded = load_onnx(dest_path, domain=domain_spec)
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid ONNX model: {exc}") from exc

        uploaded_paths[model_id] = dest_path
        if domain_spec is not None:
            uploaded_domains[model_id] = domain_spec
        model_cache[model_id] = loaded

        return {
            "model_id": model_id,
            "metadata": loaded.metadata.model_dump(mode="json"),
        }

    @app.post("/api/decompile")
    def decompile(req: DecompileRequest) -> dict[str, Any]:
        loaded = _load_model_entry(req.model_id)
        cfg = req.config or DecompileConfig(
            task_type=loaded.metadata.domain.task_type,
            verify_smt=loaded.metadata.is_piecewise_linear,
            max_degree=1 if loaded.metadata.is_piecewise_linear else 2,
        )
        run_id = f"{req.model_id}_{uuid.uuid4().hex[:8]}"
        result = run_decompile_pipeline(
            loaded,
            cfg,
            run_benchmarks_flag=req.run_benchmarks,
            run_pareto_flag=req.run_pareto,
            run_id=run_id,
        )
        run_cache[result.run_id] = {"result": result, "model": loaded}
        return result.model_dump(mode="json")

    @app.post("/api/decompile/stream")
    async def decompile_stream(req: DecompileRequest) -> StreamingResponse:
        loaded = _load_model_entry(req.model_id)
        cfg = req.config or DecompileConfig(
            task_type=loaded.metadata.domain.task_type,
            verify_smt=loaded.metadata.is_piecewise_linear,
            max_degree=1 if loaded.metadata.is_piecewise_linear else 2,
        )
        run_id = f"{req.model_id}_{uuid.uuid4().hex[:8]}"
        q: Queue[tuple[str, Any] | None] = Queue()

        def _on_iter(it: CEGISIteration) -> None:
            q.put(("iteration", it.model_dump(mode="json")))

        def _worker() -> None:
            try:
                result = run_decompile_pipeline(
                    loaded,
                    cfg,
                    run_benchmarks_flag=req.run_benchmarks,
                    run_pareto_flag=req.run_pareto,
                    on_iteration=_on_iter,
                    run_id=run_id,
                )
                run_cache[result.run_id] = {"result": result, "model": loaded}
                q.put(("complete", result.model_dump(mode="json")))
            except Exception as exc:  # noqa: BLE001 - stream worker error forwarded to SSE client
                q.put(("error", {"detail": str(exc)}))
            finally:
                q.put(None)

        threading.Thread(target=_worker, daemon=True).start()

        async def _event_gen():
            while True:
                item = await asyncio.to_thread(q.get)
                if item is None:
                    break
                ev, payload = item
                yield _sse_frame(ev, payload)

        return StreamingResponse(
            _event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        result, _ = _load_run_entry(run_id)
        return result.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/bundle")
    def get_run_bundle(run_id: str) -> dict[str, Any]:
        result, model_obj = _load_run_entry(run_id)
        stem = _sanitize_id(result.metadata.name or run_id)
        packaged = resolved_bundles / f"{stem}.gbundle"
        packaged_cert = packaged / "certificate.json"
        use_packaged = (
            run_id == stem and packaged_cert.is_file()
            and json.loads(packaged_cert.read_text(encoding="utf-8")) == result.certificate.model_dump(mode="json")
        )
        bundle_dir = packaged if use_packaged else upload_dir / f"{_sanitize_id(run_id)}.gbundle"
        if not use_packaged and model_obj is not None:
            from glassbox.certificate import write_artifact_bundle

            with bundle_lock:
                if not (bundle_dir / "manifest.json").is_file():
                    write_artifact_bundle(result, model_obj.path, bundle_dir)

        manifest_path = bundle_dir / "manifest.json"
        manifest_data: dict[str, Any] | None = None
        verification_data: dict[str, Any] | None = None
        if manifest_path.is_file():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                from glassbox.certificate import verify_artifact_bundle

                report = verify_artifact_bundle(bundle_dir, recompute_smt=False)
                verification_data = report.model_dump(mode="json")
            except ImportError:
                pass

        return {
            "run_id": result.run_id,
            "model_name": result.metadata.name,
            "bundle_available": manifest_data is not None,
            "bundle_path": str(bundle_dir) if manifest_data else None,
            "cli_verify_command": f'glassbox verify "{bundle_dir}"' if manifest_data else None,
            "manifest": manifest_data,
            "verification": verification_data,
        }

    @app.post("/api/explain")
    def explain_point(req: ExplainRequest) -> dict[str, Any]:
        result, model_obj = _load_run_entry(req.run_id)
        _check_domain_point(result.metadata.domain, req.x)
        if model_obj is None:
            raise HTTPException(404, "Matching source ONNX model is unavailable for this run")
        from glassbox.loader import explain_decision

        proof = result.certificate.real_arithmetic_proof
        exp = explain_decision(
            model_obj, result.program, req.x,
            leaf_proven_map=proof.leaf_proven_map if proof else None,
        )
        return exp.model_dump(mode="json")

    @app.post("/api/slice")
    def slice_space(req: SliceRequest) -> dict[str, Any]:
        result, model_obj = _load_run_entry(req.run_id)
        if model_obj is None:
            raise HTTPException(
                status_code=404,
                detail=f"Source ONNX model for run '{req.run_id}' is not available",
            )
        return compute_slice_payload(
            model_obj,
            result,
            dim_x=req.dim_x,
            dim_y=req.dim_y,
            fixed_values=req.fixed_values,
            resolution=req.resolution,
        )

    @app.get("/api/runs/{run_id}/download/{target}")
    def download_artifact(run_id: str, target: str) -> Response:
        result, _ = _load_run_entry(run_id)
        stem = _sanitize_id(result.metadata.name or run_id)
        target_key = target.lower()

        target_map: dict[str, tuple[str, str, str]] = {
            "c": (result.code.get("c", ""), f"{stem}.c", "text/x-c"),
            "h": (result.code.get("h", ""), f"{stem}.h", "text/x-c"),
            "rust": (result.code.get("rust", ""), f"{stem}.rs", "text/plain"),
            "python": (result.code.get("python", ""), f"{stem}.py", "text/x-python"),
            "ir": (
                result.code.get("ir_json") or result.program.model_dump_json(indent=2),
                f"{stem}.ir.json",
                "application/json",
            ),
            "mermaid": (result.code.get("mermaid", ""), f"{stem}.mermaid.md", "text/markdown"),
            "certificate": (
                result.certificate.model_dump_json(indent=2),
                f"{stem}.certificate.json",
                "application/json",
            ),
        }
        if target_key not in target_map:
            raise HTTPException(status_code=404, detail=f"Unknown artifact target: {target}")
        content, filename, media_type = target_map[target_key]
        if not content:
            raise HTTPException(status_code=404, detail=f"Artifact '{target}' not generated")
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if resolved_studio.is_dir():
        app.mount("/studio", StaticFiles(directory=str(resolved_studio), html=True), name="studio")

    if resolved_site.is_dir():
        app.mount("/", StaticFiles(directory=str(resolved_site), html=True), name="site")
    else:

        @app.get("/", include_in_schema=False)
        def _redirect_root() -> RedirectResponse:
            return RedirectResponse(url="/studio/")

    return app
