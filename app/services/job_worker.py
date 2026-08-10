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
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.models.database import Job, async_session, utc_now
from app.services.detector import get_detector
from app.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)
settings = get_settings()

_QUEUE: asyncio.Queue[str] | None = None
_WORKER_TASK: asyncio.Task[None] | None = None
_JOB_RUNNING = False
_STOPPING = False


class JobPersistenceError(RuntimeError):
    """A job result could not be committed to durable state."""


def get_queue() -> asyncio.Queue[str]:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = asyncio.Queue(maxsize=settings.max_pending_jobs)
    return _QUEUE


def worker_is_running() -> bool:
    return _WORKER_TASK is not None and not _WORKER_TASK.done()


def gsr_json_path(job_id: str) -> str:
    """Return the server-owned JSON path for a UUID job identifier."""
    canonical_job_id = str(UUID(job_id))
    return os.path.join(settings.artifact_dir, f"job_{canonical_job_id}_gsr.json")


def build_output_video_path(artifact_dir: str, job_id: str) -> str:
    """Return the server-owned output path for a UUID job identifier."""
    canonical_job_id = str(UUID(job_id))
    return os.path.join(artifact_dir, f"job_{canonical_job_id}_output.mp4")


def partial_gsr_json_path(job_id: str) -> str:
    return f"{gsr_json_path(job_id)}.part"


async def _update_job(job_id: str, **fields) -> None:
    for attempt in range(3):
        try:
            async with async_session() as db:
                result = await db.execute(select(Job).where(Job.job_id == job_id))
                job = result.scalar_one_or_none()
                if job is None:
                    logger.warning("job update skipped: id=%s not found", job_id)
                    return
                for key, value in fields.items():
                    setattr(job, key, value)
                await db.commit()
                return
        except SQLAlchemyError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.1 * (2**attempt))


async def _load_job(job_id: str) -> Job | None:
    async with async_session() as db:
        result = await db.execute(select(Job).where(Job.job_id == job_id))
        return result.scalar_one_or_none()


async def _update_progress(job_id: str, stage: str, progress: float) -> None:
    """Update only a running job, ignoring callbacks that arrive late."""
    async with async_session() as db:
        await db.execute(
            update(Job)
            .where(Job.job_id == job_id, Job.status == "running")
            .values(stage=stage, progress=float(progress))
        )
        await db.commit()


def _build_processor() -> VideoProcessor:
    """Construct the validated offline pipeline."""
    from app.services.calibration import make_calibrator
    from app.services.embedder import get_embedder
    from app.services.jersey import get_jersey_recognizer
    from app.services.keypoints import get_keypoints_extractor
    from app.services.minimap import get_minimap_renderer
    from app.services.team_classifier import make_team_classifier
    from app.services.tracker import make_tracker

    detector = get_detector()
    embedder = get_embedder()

    return VideoProcessor(
        detector=detector,
        tracker_factory=lambda fps: make_tracker(frame_rate=int(round(fps)) or 30),
        jersey_recognizer=get_jersey_recognizer(),
        team_classifier_factory=make_team_classifier,
        keypoints_extractor=get_keypoints_extractor(),
        calibrator_factory=make_calibrator,
        minimap_renderer=get_minimap_renderer(),
        embedder=embedder,
    )


def remove_artifact(path: str | Path | None) -> bool:
    """Best-effort delete of an upload or generated artifact."""
    if not path:
        return True
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not delete artifact %s: %s", path, exc)
        return False
    return True


def _run_processing_blocking(
    job_id: str,
    input_path: str,
    output_path: str,
    main_loop: asyncio.AbstractEventLoop,
    source_filename: str | None = None,
) -> None:
    """Run inference in an executor and publish progress to the event loop."""
    processor = _build_processor()

    def _progress(stage: str, pct: float) -> None:
        future = asyncio.run_coroutine_threadsafe(
            _update_progress(job_id, stage, float(pct)),
            main_loop,
        )

        def _log_failure(done) -> None:
            try:
                done.result()
            except Exception:
                logger.debug("progress update failed", exc_info=True)

        future.add_done_callback(_log_failure)

    _, state = processor.process_video(
        input_path=input_path,
        output_path=output_path,
        source_filename=source_filename,
        progress_cb=_progress,
    )

    json_path = gsr_json_path(job_id)
    partial_path = partial_gsr_json_path(job_id)
    try:
        with open(partial_path, "w", encoding="utf-8") as file:
            json.dump(state.to_gsr_json(), file)
        os.replace(partial_path, json_path)
    finally:
        remove_artifact(partial_path)


async def _process_job(job_id: str) -> None:
    output_path = build_output_video_path(settings.artifact_dir, job_id)
    job: Job | None = None
    processing_completed = False

    try:
        job = await _load_job(job_id)
        if job is None:
            logger.warning("job skipped: id=%s not found", job_id)
            return
        if job.status not in ("queued", "running"):
            logger.warning("job skipped: id=%s status=%s", job_id, job.status)
            return
        if not job.input_path or not os.path.isfile(job.input_path):
            raise FileNotFoundError("input video is missing")

        await _update_job(
            job_id,
            status="running",
            started_at=utc_now(),
            stage="pass1",
            progress=0.0,
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _run_processing_blocking,
            job_id,
            job.input_path,
            output_path,
            loop,
            job.input_filename,
        )
        processing_completed = True
        await _update_job(
            job_id,
            status="done",
            stage="done",
            progress=100.0,
            output_video_path=output_path,
            finished_at=utc_now(),
        )
    except Exception as exc:
        logger.exception("job failed: id=%s", job_id)
        if processing_completed:
            remove_artifact(job.input_path if job is not None else None)
            raise JobPersistenceError(f"completed job {job_id} could not be committed") from exc

        try:
            await _update_job(
                job_id,
                status="error",
                stage=None,
                progress=None,
                output_video_path=None,
                error_message=f"{type(exc).__name__}: {exc}",
                finished_at=utc_now(),
            )
        except SQLAlchemyError as persistence_error:
            logger.exception("could not persist failure state for job %s", job_id)
            remove_artifact(output_path)
            remove_artifact(gsr_json_path(job_id))
            remove_artifact(partial_gsr_json_path(job_id))
            raise JobPersistenceError(
                f"failed job {job_id} could not be committed"
            ) from persistence_error
        remove_artifact(job.input_path if job is not None else None)
        remove_artifact(output_path)
        remove_artifact(gsr_json_path(job_id))
        remove_artifact(partial_gsr_json_path(job_id))
        return

    remove_artifact(job.input_path)


async def _worker_loop() -> None:
    global _JOB_RUNNING
    queue = get_queue()
    logger.info("job worker started")
    while True:
        job_id = await queue.get()
        _JOB_RUNNING = True
        try:
            await _process_job(job_id)
        except asyncio.CancelledError:
            raise
        except JobPersistenceError:
            logger.exception("job worker stopped after a persistence failure")
            raise
        except Exception:
            logger.exception("unhandled worker error: id=%s", job_id)
        finally:
            _JOB_RUNNING = False
            queue.task_done()
        if _STOPPING:
            return


async def _recover_stale_jobs() -> None:
    """Restore queued or interrupted jobs and finalize complete artifacts."""
    async with async_session() as db:
        result = await db.execute(select(Job).where(Job.status.in_(("queued", "running"))))
        stale = result.scalars().all()
        requeued = 0
        recovered = 0
        failed = 0
        for job in stale:
            if job.status == "running":
                output_path = build_output_video_path(settings.artifact_dir, job.job_id)
                json_path = gsr_json_path(job.job_id)
                if os.path.isfile(output_path) and os.path.isfile(json_path):
                    job.status = "done"
                    job.stage = "done"
                    job.progress = 100.0
                    job.output_video_path = output_path
                    job.error_message = None
                    job.finished_at = utc_now()
                    remove_artifact(job.input_path)
                    recovered += 1
                    continue

                remove_artifact(output_path)
                remove_artifact(json_path)
                remove_artifact(partial_gsr_json_path(job.job_id))
                job.status = "queued"
                job.stage = None
                job.progress = None
                job.started_at = None
                job.finished_at = None
                job.output_video_path = None
                job.error_message = None

            if not job.input_path or not os.path.isfile(job.input_path):
                job.status = "error"
                job.stage = None
                job.progress = None
                job.output_video_path = None
                job.error_message = "input video is missing after service restart"
                job.finished_at = utc_now()
                failed += 1
            else:
                try:
                    await enqueue_job(job.job_id)
                    requeued += 1
                except asyncio.QueueFull:
                    job.status = "error"
                    job.stage = None
                    job.progress = None
                    job.output_video_path = None
                    job.error_message = "queue capacity exceeded during service restart"
                    job.finished_at = utc_now()
                    remove_artifact(job.input_path)
                    failed += 1
        await db.commit()
    if requeued or recovered or failed:
        logger.info(
            "job recovery: requeued=%d recovered=%d failed=%d",
            requeued,
            recovered,
            failed,
        )


async def start_worker() -> None:
    global _STOPPING, _WORKER_TASK
    if _WORKER_TASK is not None and not _WORKER_TASK.done():
        return
    _STOPPING = False
    await _recover_stale_jobs()
    _WORKER_TASK = asyncio.create_task(_worker_loop(), name="gsr-worker")


async def stop_worker() -> None:
    global _QUEUE, _STOPPING, _WORKER_TASK
    if _WORKER_TASK is None:
        return
    _STOPPING = True
    if not _JOB_RUNNING and not _WORKER_TASK.done():
        # Video inference runs in an executor and cannot be force-cancelled;
        # an active job is left to commit its terminal state instead. Queued
        # jobs remain in SQLite and are recovered on the next start.
        _WORKER_TASK.cancel()
    try:
        await _WORKER_TASK
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("job worker stopped with an error")
    _WORKER_TASK = None
    _QUEUE = None


async def enqueue_job(job_id: str) -> None:
    """Enqueue without waiting so API requests can reject overload promptly."""
    get_queue().put_nowait(job_id)
