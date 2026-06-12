"""Shared upload validation/saving for the /forward endpoints."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import get_settings

settings = get_settings()

_CHUNK = 1024 * 1024


def validate_video(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="no filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_video_extensions:
        raise HTTPException(status_code=400, detail=f"unsupported extension {ext}")
    if file.content_type and not (
        file.content_type.startswith("video/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400, detail=f"unsupported content_type {file.content_type}"
        )


async def save_upload(file: UploadFile, job_id: str) -> str:
    """Stream the upload to temp_dir, enforcing the size limit as we go
    (so an oversized body is rejected without buffering it in memory)."""
    os.makedirs(settings.temp_dir, exist_ok=True)
    ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    path = os.path.join(settings.temp_dir, f"input_{job_id}{ext}")
    limit = settings.max_video_size_mb * 1024 * 1024
    written = 0
    try:
        with open(path, "wb") as f:
            while chunk := await file.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=400,
                        detail=f"file too large: > {settings.max_video_size_mb} MB",
                    )
                f.write(chunk)
    except HTTPException:
        os.unlink(path)
        raise
    return path
