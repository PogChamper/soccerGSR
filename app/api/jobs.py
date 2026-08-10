"""Async GSR job endpoints.

POST   /jobs               accept a video, enqueue a job, return 202 + job_id.
GET    /jobs/{id}          job status (queued/running/done/error) + progress.
GET    /jobs/{id}/video    download annotated mp4.
GET    /jobs/{id}/gsr.json per-clip GSR state JSON.
DELETE /jobs/{id}          delete a finished job and its artifacts.
GET    /jobs               list recent jobs.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.uploads import save_upload, upload_name, validate_video
from app.config import get_settings
from app.models.database import Job, get_db, utc_now
from app.services.job_worker import (
    build_output_video_path,
    enqueue_job,
    gsr_json_path,
    partial_gsr_json_path,
    remove_artifact,
)

router = APIRouter()
settings = get_settings()
JobStatus = Literal["queued", "running", "done", "error"]


class JobLinks(BaseModel):
    self: str | None = None
    video: str | None = None
    gsr_json: str | None = None


class JobSubmission(BaseModel):
    job_id: str
    status: JobStatus
    links: JobLinks


class JobDetails(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None
    progress: float | None
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    input_filename: str | None
    error_message: str | None
    links: JobLinks


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None
    progress: float | None
    created_at: datetime | None
    input_filename: str | None
    error_message: str | None


class JobPage(BaseModel):
    items: list[JobSummary]
    limit: int
    offset: int


def _api_timestamp(value: datetime | None) -> datetime | None:
    """Attach UTC to timestamps read back as naive values from SQLite."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC)


async def _get_job_or_404(job_id: str, db: AsyncSession) -> Job:
    res = await db.execute(select(Job).where(Job.job_id == job_id))
    job = res.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


async def _delete_job_record(job_id: str, db: AsyncSession) -> None:
    """Remove a committed job that could not be admitted to the worker queue."""
    await db.rollback()
    await db.execute(delete(Job).where(Job.job_id == job_id))
    await db.commit()


@router.post("/jobs", status_code=202, response_model=JobSubmission)
async def create_job(
    video: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
    db: AsyncSession = Depends(get_db),
):
    """Enqueue a video for GSR processing; poll GET /jobs/{job_id} for status."""
    validate_video(video)
    job_id = str(uuid4())
    input_path: str | None = None
    record_added = False
    admitted = False
    try:
        input_path = await save_upload(video, job_id)
        job = Job(
            job_id=job_id,
            created_at=utc_now(),
            status="queued",
            input_filename=upload_name(video),
            input_path=input_path,
        )
        db.add(job)
        record_added = True
        await db.commit()

        await enqueue_job(job_id)
        admitted = True
    except asyncio.QueueFull as exc:
        if record_added:
            await _delete_job_record(job_id, db)
        else:
            await db.rollback()
        raise HTTPException(
            status_code=503,
            detail="job queue is full",
            headers={"Retry-After": "5"},
        ) from exc
    except BaseException:
        await db.rollback()
        if record_added:
            await _delete_job_record(job_id, db)
        raise
    finally:
        if not admitted:
            remove_artifact(input_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "links": {
            "self": f"/jobs/{job_id}",
            "video": f"/jobs/{job_id}/video",
            "gsr_json": f"/jobs/{job_id}/gsr.json",
        },
    }


@router.get("/jobs/{job_id}", response_model=JobDetails)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await _get_job_or_404(job_id, db)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "created_at": _api_timestamp(job.created_at),
        "started_at": _api_timestamp(job.started_at),
        "finished_at": _api_timestamp(job.finished_at),
        "input_filename": job.input_filename,
        "error_message": job.error_message,
        "links": {
            "self": f"/jobs/{job_id}",
            "video": f"/jobs/{job_id}/video" if job.status == "done" else None,
            "gsr_json": f"/jobs/{job_id}/gsr.json" if job.status == "done" else None,
        },
    }


@router.get("/jobs/{job_id}/video")
async def get_job_video(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await _get_job_or_404(job_id, db)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job not done (status={job.status})")
    if not job.output_video_path or not os.path.exists(job.output_video_path):
        raise HTTPException(status_code=410, detail="output video missing on disk")
    return FileResponse(
        job.output_video_path,
        media_type="video/mp4",
        filename=f"gsr_{job_id}.mp4",
    )


@router.get("/jobs/{job_id}/gsr.json")
async def get_job_gsr_json(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await _get_job_or_404(job_id, db)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job not done (status={job.status})")
    path = gsr_json_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="gsr.json missing on disk")
    return FileResponse(path, media_type="application/json")


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """Delete a completed/failed job and its local artifacts."""
    job = await _get_job_or_404(job_id, db)
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"job is active (status={job.status})")

    artifacts = {
        job.input_path,
        job.output_video_path,
        build_output_video_path(settings.artifact_dir, job_id),
        gsr_json_path(job_id),
        partial_gsr_json_path(job_id),
    }
    failed = [path for path in artifacts if not remove_artifact(path)]
    if failed:
        raise HTTPException(status_code=500, detail="could not remove all job artifacts")
    await db.delete(job)
    await db.commit()
    return Response(status_code=204)


@router.get("/jobs", response_model=JobPage)
async def list_jobs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: JobStatus | None = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Job).order_by(desc(Job.created_at))
    if status:
        q = q.where(Job.status == status)
    q = q.offset(offset).limit(limit)
    res = await db.execute(q)
    jobs = res.scalars().all()
    return {
        "items": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "stage": j.stage,
                "progress": j.progress,
                "created_at": _api_timestamp(j.created_at),
                "input_filename": j.input_filename,
                "error_message": j.error_message,
            }
            for j in jobs
        ],
        "limit": limit,
        "offset": offset,
    }
