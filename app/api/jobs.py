"""Async GSR job endpoints.

POST /forward            — accept a video, enqueue a job, return 202 + job_id.
GET  /jobs/{id}          — job status (queued/running/done/error) + progress.
GET  /jobs/{id}/video    — download annotated mp4.
GET  /jobs/{id}/gsr.json — per-clip GSR state JSON.
GET  /jobs               — list recent jobs.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.uploads import save_upload, validate_video
from app.models.database import Job, get_db
from app.services.job_worker import enqueue_job, gsr_json_path

router = APIRouter()


async def _get_job_or_404(job_id: str, db: AsyncSession) -> Job:
    res = await db.execute(select(Job).where(Job.job_id == job_id))
    job = res.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post("/forward", status_code=202)
async def forward_async(
    image: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
    db: AsyncSession = Depends(get_db),
):
    """Enqueue a video for GSR processing; poll GET /jobs/{job_id} for status."""
    validate_video(image)
    job_id = str(uuid4())
    input_path = await save_upload(image, job_id)

    job = Job(
        job_id=job_id,
        created_at=datetime.utcnow(),
        status="queued",
        input_filename=image.filename,
        input_path=input_path,
    )
    db.add(job)
    await db.commit()

    await enqueue_job(job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "links": {
            "self": f"/jobs/{job_id}",
            "video": f"/jobs/{job_id}/video",
            "gsr_json": f"/jobs/{job_id}/gsr.json",
        },
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await _get_job_or_404(job_id, db)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "input_filename": job.input_filename,
        "error_message": job.error_message,
        "links": {
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


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
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
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "input_filename": j.input_filename,
                "error_message": j.error_message,
            }
            for j in jobs
        ],
        "limit": limit,
        "offset": offset,
    }
