"""Single-worker async job runner for the GSR pipeline.

One worker because each job is GPU-bound: parallel jobs would compete for
VRAM and OOM with all models loaded. The blocking pipeline runs in a thread
executor so uvicorn keeps serving HTTP; the queue itself is in-memory, so
stale DB jobs are reconciled on startup (see ``_recover_stale_jobs``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.config import get_settings
from app.models.database import Job, async_session
from app.services import history_service
from app.services.detector import get_detector
from app.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)
settings = get_settings()

_QUEUE: Optional[asyncio.Queue[str]] = None
_WORKER_TASK: Optional[asyncio.Task] = None


def get_queue() -> asyncio.Queue[str]:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = asyncio.Queue()
    return _QUEUE


def gsr_json_path(job_id: str) -> str:
    """Where a job's GSR JSON lives on disk (kept out of the DB: it can be
    tens of MB per clip)."""
    return os.path.join(settings.temp_dir, f"job_{job_id}_gsr.json")


# --------------------------------------------------------------- DB helpers


async def _update_job(job_id: str, **fields) -> None:
    async with async_session() as db:
        result = await db.execute(select(Job).where(Job.job_id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            logger.warning(f"_update_job: job_id={job_id} not found")
            return
        for k, v in fields.items():
            setattr(job, k, v)
        await db.commit()


async def _load_job(job_id: str) -> Optional[Job]:
    async with async_session() as db:
        result = await db.execute(select(Job).where(Job.job_id == job_id))
        return result.scalar_one_or_none()


# ------------------------------------------------------------- worker body


def _build_processor() -> VideoProcessor:
    """Construct VideoProcessor with all available phases wired up.
    Missing optional models degrade gracefully (with a warning)."""
    from app.services.embedder import get_embedder
    from app.services.jersey import get_jersey_recognizer
    from app.services.team_classifier import make_team_classifier
    from app.services.tracker import make_tracker

    detector = get_detector()
    try:
        embedder = get_embedder()
    except Exception as exc:
        logger.warning(f"Embedder disabled (DINOv3 ONNX unavailable: {exc})")
        embedder = None

    keypoints_extractor = None
    calibrator_factory = None
    minimap_renderer = None
    try:
        from app.services.calibration import make_calibrator
        from app.services.keypoints import get_keypoints_extractor
        from app.services.minimap import get_minimap_renderer

        keypoints_extractor = get_keypoints_extractor()
        calibrator_factory = make_calibrator
        minimap_renderer = get_minimap_renderer()
    except Exception as exc:
        logger.warning(f"Calibration/minimap disabled: {exc}")

    return VideoProcessor(
        detector=detector,
        tracker_factory=lambda fps: make_tracker(
            frame_rate=int(round(fps)) or 30,
            with_reid=embedder is not None,
        ),
        jersey_recognizer=get_jersey_recognizer(),
        team_classifier_factory=make_team_classifier,
        keypoints_extractor=keypoints_extractor,
        calibrator_factory=calibrator_factory,
        minimap_renderer=minimap_renderer,
        embedder=embedder,
    )


def _run_processing_blocking(
    job_id: str,
    input_path: str,
    output_path: str,
    main_loop: asyncio.AbstractEventLoop,
) -> dict:
    """Executor body. Progress callbacks are scheduled back onto the main
    event loop via ``run_coroutine_threadsafe``."""
    processor = _build_processor()

    def _progress(stage: str, pct: float):
        try:
            asyncio.run_coroutine_threadsafe(
                _update_job(job_id, stage=stage, progress=float(pct)),
                main_loop,
            )
        except Exception:
            logger.debug("progress update failed", exc_info=True)

    out_path, state, stats = processor.process_video(
        input_path=input_path,
        output_path=output_path,
        progress_cb=_progress,
    )

    with open(gsr_json_path(job_id), "w", encoding="utf-8") as f:
        json.dump(state.to_gsr_json(), f)

    return {
        "out_path": out_path,
        "stats": {
            "processing_time": stats.processing_time,
            "pass1_time": stats.pass1_time,
            "aggregate_time": stats.aggregate_time,
            "pass2_time": stats.pass2_time,
            "frames_processed": stats.frames_processed,
            "total_detections": stats.total_detections,
            "players_count": stats.players_count,
            "goalkeepers_count": stats.goalkeepers_count,
            "referees_count": stats.referees_count,
            "balls_count": stats.balls_count,
            "n_tracks": stats.n_tracks,
            "n_tracks_with_jersey": stats.n_tracks_with_jersey,
            "n_frames_calibrated": stats.n_frames_calibrated,
        },
        "meta": {
            "filename": state.meta.filename,
            "width": state.meta.width,
            "height": state.meta.height,
            "fps": state.meta.fps,
            "frame_count": state.meta.frame_count,
            "duration": state.meta.duration,
            "size_mb": state.meta.size_mb,
        },
    }


def _cleanup_input(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError as exc:
            logger.warning(f"could not delete input file {path}: {exc}")


async def _process_job(job_id: str) -> None:
    job = await _load_job(job_id)
    if job is None:
        logger.warning(f"_process_job: job {job_id} not found, skipping")
        return
    if job.status not in ("queued", "running"):
        logger.warning(f"_process_job: job {job_id} is in status={job.status}, skipping")
        return

    await _update_job(
        job_id,
        status="running",
        started_at=datetime.utcnow(),
        stage="pass1",
        progress=0.0,
    )

    output_path = os.path.join(
        settings.temp_dir,
        f"job_{job_id}_{job.input_filename or 'output'}",
    )
    if not output_path.endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            _run_processing_blocking,
            job_id,
            job.input_path,
            output_path,
            loop,
        )
    except Exception as exc:
        # full traceback goes to the log; the DB (and thus the API) only gets
        # a short message — no internal paths/stack frames for clients
        logger.exception(f"job {job_id} failed")
        await _update_job(
            job_id,
            status="error",
            stage=None,
            progress=None,
            error_message=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.utcnow(),
        )
        async with async_session() as db:
            await history_service.create_history_entry(
                db=db,
                status="error",
                error_message=str(exc),
                input_filename=job.input_filename,
            )
        _cleanup_input(job.input_path)
        return

    stats = result["stats"]
    meta = result["meta"]
    await _update_job(
        job_id,
        status="done",
        stage="done",
        progress=100.0,
        output_video_path=result["out_path"],
        finished_at=datetime.utcnow(),
    )

    async with async_session() as db:
        await history_service.create_history_entry(
            db=db,
            status="success",
            input_filename=meta["filename"],
            input_size_mb=meta["size_mb"],
            input_width=meta["width"],
            input_height=meta["height"],
            input_duration=meta["duration"],
            input_fps=meta["fps"],
            input_frames=meta["frame_count"],
            processing_time=stats["processing_time"],
            frames_processed=stats["frames_processed"],
            total_detections=stats["total_detections"],
            players_count=stats["players_count"],
            goalkeepers_count=stats["goalkeepers_count"],
            referees_count=stats["referees_count"],
            balls_count=stats["balls_count"],
        )
    _cleanup_input(job.input_path)


async def _worker_loop() -> None:
    queue = get_queue()
    logger.info("job worker started")
    while True:
        job_id = await queue.get()
        try:
            await _process_job(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"unhandled error processing {job_id}")
        finally:
            queue.task_done()


async def _recover_stale_jobs() -> None:
    """The in-memory queue does not survive a restart: mark interrupted
    ``running`` jobs as error, re-enqueue ``queued`` ones."""
    async with async_session() as db:
        result = await db.execute(
            select(Job).where(Job.status.in_(("queued", "running")))
        )
        stale = result.scalars().all()
        requeued = 0
        failed = 0
        for job in stale:
            if job.status == "running":
                job.status = "error"
                job.stage = None
                job.progress = None
                job.error_message = "interrupted by service restart"
                job.finished_at = datetime.utcnow()
                failed += 1
            else:
                await get_queue().put(job.job_id)
                requeued += 1
        await db.commit()
    if requeued or failed:
        logger.info(
            f"job recovery: re-enqueued {requeued} queued job(s), "
            f"marked {failed} interrupted running job(s) as error"
        )


async def start_worker() -> None:
    global _WORKER_TASK
    if _WORKER_TASK is not None and not _WORKER_TASK.done():
        return
    await _recover_stale_jobs()
    _WORKER_TASK = asyncio.create_task(_worker_loop(), name="gsr-worker")


async def stop_worker() -> None:
    global _WORKER_TASK
    if _WORKER_TASK is None:
        return
    _WORKER_TASK.cancel()
    try:
        await _WORKER_TASK
    except (asyncio.CancelledError, Exception):
        pass
    _WORKER_TASK = None


async def enqueue_job(job_id: str) -> None:
    await get_queue().put(job_id)
