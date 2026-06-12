import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, forward, history, jobs, stats
from app.config import get_settings
from app.models.database import init_db
from app.services.job_worker import start_worker, stop_worker
from app.utils.cuda_env import bootstrap as cuda_bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("app").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
settings = get_settings()


def ensure_models() -> None:
    """Best-effort: download / verify all registered models on startup."""
    from app.utils.models_registry import REGISTRY, ensure_models as _em

    if not settings.model_auto_download:
        logger.info("model_auto_download=False, skipping registry ensure")
        return

    results = _em(list(REGISTRY), skip_missing_remotes=True)
    for name, path in results.items():
        if path and Path(path).exists():
            logger.info(f"model OK : {name:20s} -> {path}")
        else:
            logger.warning(
                f"model MISS: {name:20s} (services depending on it will fail loudly later)"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}...")
    if "change-in-production" in settings.secret_key:
        logger.warning(
            "SECRET_KEY is the default placeholder — set a real one in .env "
            "before exposing this service"
        )
    os.makedirs(settings.temp_dir, exist_ok=True)

    cuda_ok = cuda_bootstrap()
    logger.info(f"CUDA bootstrap: {'OK (GPU)' if cuda_ok else 'CPU fallback'}")

    ensure_models()

    await init_db()
    logger.info("Database initialized")

    # IMPORTANT: load ALL onnxruntime-gpu sessions BEFORE anything that imports
    # torch (boxmot/PnLCalib). torch's cudnn frontend init conflicts with ORT
    # cudnn frontend on CUDA13 + cudnn 9.1, causing CUDNN_BACKEND_API_FAILED on
    # session creation if torch wins the race. Load order matters here.
    try:
        from app.services.detector import get_detector

        get_detector()
        logger.info(f"Detector loaded (backend={settings.detector_backend})")
    except Exception as exc:
        logger.exception(f"Could not pre-load detector: {exc}")

    try:
        from app.services.jersey import get_jersey_recognizer

        get_jersey_recognizer()
        logger.info("Jersey recognizer loaded (visibility gate + OCR)")
    except Exception as exc:
        logger.warning(f"Could not pre-load jersey recognizer: {exc}")

    try:
        from app.services.embedder import get_embedder

        get_embedder()
        logger.info("DINOv3 embedder loaded (ReID + team clustering)")
    except Exception as exc:
        logger.warning(f"Could not pre-load DINOv3 embedder: {exc}")

    try:
        from app.services.keypoints import get_keypoints_extractor

        get_keypoints_extractor()
        logger.info("HRNet keypoints + lines extractor loaded")
    except Exception as exc:
        logger.warning(f"Could not pre-load keypoints extractor: {exc}")

    await start_worker()
    logger.info("GSR job worker started")

    yield

    logger.info("Stopping job worker...")
    await stop_worker()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="""
## SoccerGSR ML Service (async)

Two-pass GSR pipeline (detection [+tracking +jersey +team +calibration]) for
soccer broadcast clips. All inference is GPU-bound; one job at a time.

### Inference
- **POST /forward** — submit a video, returns `202 Accepted` + `job_id`.
- **GET /jobs/{id}** — poll status / progress.
- **GET /jobs/{id}/video** — download annotated mp4 (when `done`).
- **GET /jobs/{id}/gsr.json** — structured per-frame GSR state.
- **POST /forward/sync** — legacy sync wrapper (polls until done).

### Other
- **GET /history**, **DELETE /history**, **GET /stats**
- **POST /auth/...** — JWT registration / login
""",
    version="2.0.0",
    lifespan=lifespan,
)

# Auth uses bearer tokens (no cookies), so credentials are not needed —
# wildcard origins + allow_credentials=True is an unsafe combination.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, tags=["gsr"])
app.include_router(forward.router, tags=["gsr-legacy"])
app.include_router(history.router, tags=["history"])
app.include_router(stats.router, tags=["statistics"])
app.include_router(auth.router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": settings.app_name,
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "forward": "POST /forward (202 + job_id)",
            "job_status": "GET /jobs/{id}",
            "job_video": "GET /jobs/{id}/video",
            "job_gsr_json": "GET /jobs/{id}/gsr.json",
            "list_jobs": "GET /jobs",
            "forward_sync": "POST /forward/sync (legacy)",
            "history": "GET /history",
            "delete_history": "DELETE /history (admin JWT)",
            "stats": "GET /stats",
            "auth": {
                "register": "POST /auth/register",
                "login": "POST /auth/login",
                "me": "GET /auth/me",
            },
        },
    }


@app.get("/health", tags=["root"])
async def health_check():
    model_status = "unknown"
    try:
        from app.services.detector import get_detector

        get_detector()
        model_status = "loaded"
    except Exception as exc:
        model_status = f"error: {exc}"

    cuda_ok = False
    try:
        import onnxruntime as ort

        cuda_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass

    return {
        "status": "healthy",
        "model": model_status,
        "cuda": cuda_ok,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
