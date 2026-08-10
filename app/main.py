import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import jobs
from app.config import get_settings
from app.models.database import close_db, init_db
from app.services.job_worker import start_worker, stop_worker, worker_is_running
from app.utils.cuda_env import bootstrap as cuda_bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.startup_error = None
    worker_started = False
    database_started = False
    try:
        logger.info("Starting %s", settings.app_name)
        os.makedirs(settings.artifact_dir, exist_ok=True)

        cuda_ok = cuda_bootstrap()
        logger.info("CUDA bootstrap: %s", "GPU" if cuda_ok else "CPU")

        await init_db()
        database_started = True
        logger.info("Database initialized")

        from app.services.detector import get_detector

        get_detector()
        logger.info("DEIMv2 detector loaded")

        from app.services.jersey import get_jersey_recognizer

        get_jersey_recognizer()
        logger.info("Jersey recognizer loaded (visibility gate + OCR)")

        from app.services.embedder import get_embedder

        get_embedder()
        logger.info("SoccerNet OSNet loaded")

        from app.services.keypoints import get_keypoints_extractor

        get_keypoints_extractor()
        logger.info("HRNet keypoints and lines loaded")

        await start_worker()
        worker_started = True
        app.state.ready = True
        logger.info("GSR job worker started")
        yield
    except Exception as exc:
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Service lifecycle failed")
        raise
    finally:
        app.state.ready = False
        try:
            if worker_started:
                logger.info("Stopping job worker...")
                await stop_worker()
        finally:
            if database_started:
                await close_db()
        logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Local, asynchronous game-state reconstruction for soccer video. "
        "Run one Uvicorn worker per GPU process."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(jobs.router, tags=["gsr"])


@app.get("/", tags=["root"])
async def root():
    return {
        "service": settings.app_name,
        "version": __version__,
        "status": "running",
    }


@app.get("/health/live", tags=["root"])
async def health_live():
    return {"status": "alive"}


@app.get("/health", tags=["root"])
@app.get("/health/ready", tags=["root"])
async def health_ready(request: Request):
    startup_error = getattr(request.app.state, "startup_error", None)
    ready = (
        bool(getattr(request.app.state, "ready", False))
        and worker_is_running()
        and not startup_error
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "startup_error": str(startup_error) if startup_error else None,
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug,
    )
