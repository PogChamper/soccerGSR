"""Legacy synchronous POST /forward/sync.

Enqueues a job through the async pipeline and polls until it finishes, then
returns the mp4 as a stream (X-Return-Format: stream) or base64 (default).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.uploads import save_upload, validate_video
from app.models.database import Job, async_session, get_db
from app.models.schemas import ForwardResponse
from app.services.job_worker import enqueue_job, gsr_json_path

router = APIRouter()

POLL_INTERVAL_S = 0.5
POLL_TIMEOUT_S = 60 * 30


async def _wait_for_job(job_id: str) -> Job:
    """Poll job status with a short-lived DB session per check (the request
    can stay open for many minutes; holding one session that long would pin
    a pool connection)."""
    elapsed = 0.0
    while elapsed < POLL_TIMEOUT_S:
        async with async_session() as db:
            res = await db.execute(select(Job).where(Job.job_id == job_id))
            job: Optional[Job] = res.scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=500, detail="job vanished from db")
        if job.status == "done":
            return job
        if job.status == "error":
            raise HTTPException(
                status_code=500,
                detail=f"processing failed: {job.error_message or 'unknown error'}",
            )
        await asyncio.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S
    raise HTTPException(status_code=504, detail="processing timed out")


@router.post("/forward/sync", response_model=ForwardResponse)
async def forward_sync(
    image: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
    db: AsyncSession = Depends(get_db),
    x_return_format: Optional[str] = Header(default="base64", alias="X-Return-Format"),
):
    """Legacy sync endpoint: same response shape as the original /forward."""
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

    job = await _wait_for_job(job_id)
    out_path = job.output_video_path
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=410, detail="output video missing on disk")

    with open(out_path, "rb") as f:
        video_bytes = f.read()

    metadata = {"job_id": job_id, "filename": job.input_filename}
    try:
        with open(gsr_json_path(job_id), "r", encoding="utf-8") as f:
            gsr = json.load(f)
        metadata.update({
            "frames": len(gsr.get("frames", [])),
            "n_observations": len(gsr.get("observations", [])),
            "n_tracks": len(gsr.get("tracks", {})),
        })
    except (OSError, json.JSONDecodeError):
        pass

    if x_return_format == "stream":
        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=processed_{job.input_filename}"},
        )
    return ForwardResponse(
        status="success",
        video=base64.b64encode(video_bytes).decode("utf-8"),
        metadata=metadata,
    )
