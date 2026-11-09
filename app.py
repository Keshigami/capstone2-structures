"""FastAPI service on port 8102: upload an image (or click a gallery sample) and get the top-3 classes."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from structures import CLASSES_PATH, METRICS_PATH, PROJECT_DIR, SAMPLES_DIR, slug  # noqa: E402

log = logging.getLogger("structures.app")
SERVICE = "capstone2-structures"
PORT = 8102
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))


def load_samples() -> list[dict]:
    """Gallery entries: one per class, from static/samples/samples.json (written by train.py)."""
    index = SAMPLES_DIR / "samples.json"
    if index.exists():
        rows = json.loads(index.read_text())
        return [{"slug": k, "label": v["class"], "url": f"/static/samples/{v['file']}"} for k, v in rows.items()]
    # Fallback: derive from the files present.
    classes = json.loads(CLASSES_PATH.read_text())["classes"] if CLASSES_PATH.exists() else []
    by_slug = {slug(c): c for c in classes}
    return [{"slug": p.stem, "label": by_slug.get(p.stem, p.stem), "url": f"/static/samples/{p.name}"}
            for p in sorted(SAMPLES_DIR.glob("*.jpg"))]


def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.model_error = None
    try:
        from structures.inference import Predictor

        app.state.predictor = Predictor()
        log.info("model loaded: %d classes, %dpx", len(app.state.predictor.classes), app.state.predictor.image_size)
    except Exception as exc:  # noqa: BLE001 - the page must still open and say what to do
        app.state.model_error = str(exc)
        log.warning("model not loaded: %s", exc)
    yield


app = FastAPI(title="Historical structures classifier", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(PROJECT_DIR / "static")), name="static")


@app.get("/health")
def health() -> dict:
    p = app.state.predictor
    return {
        "ok": True,
        "service": SERVICE,
        "model_loaded": p is not None,
        "classes": p.classes if p else [],
        "image_size": p.image_size if p else None,
        "error": app.state.model_error,
    }


@app.get("/metrics")
def metrics():
    m = load_metrics()
    if m is None:
        raise HTTPException(status_code=404, detail="models/metrics.json not found; run `just train`")
    return m


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    m = load_metrics() or {}
    p = app.state.predictor
    ctx = {
        "service": SERVICE,
        "port": PORT,
        "model_loaded": p is not None,
        "model_error": app.state.model_error,
        "samples": load_samples(),
        "metrics": m,
        "per_class": sorted((m.get("per_class_test_accuracy") or {}).items(), key=lambda kv: -kv[1]),
        "has_curves": (PROJECT_DIR / "models" / "curves.png").exists(),
    }
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/curves.png")
def curves():
    from fastapi.responses import FileResponse

    path = PROJECT_DIR / "models" / "curves.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="models/curves.png not found; run `just train`")
    return FileResponse(path, media_type="image/png")


@app.post("/predict")
async def predict(
    file: UploadFile | None = File(None),  # noqa: B008 - FastAPI declares defaults this way
    sample: str | None = Form(None),  # noqa: B008
    top: int = 3,
):
    """Classify an uploaded image, or a gallery sample by slug (e.g. sample=gargoyle)."""
    p = app.state.predictor
    if p is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {app.state.model_error}")
    if sample:
        path = SAMPLES_DIR / f"{slug(sample)}.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"unknown sample {sample!r}")
        data, name = path.read_bytes(), path.name
    elif file is not None:
        data, name = await file.read(), file.filename or "upload"
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image larger than 20 MB")
    else:
        raise HTTPException(status_code=400, detail="send a multipart 'file' or a 'sample' name")
    try:
        result = p.predict_bytes(data, top=max(1, min(top, len(p.classes))))
    except Exception as exc:  # noqa: BLE001 - unreadable image, say so
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}") from exc
    return JSONResponse({"filename": name, **result})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)
