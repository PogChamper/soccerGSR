"""Video upload validation and persistence."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import get_settings
from app.services.job_worker import remove_artifact

settings = get_settings()

_CHUNK = 1024 * 1024


def upload_name(file: UploadFile) -> str:
    """Return a display-only basename, independent of client path syntax."""
    raw = (file.filename or "video").replace("\\", "/")
    return raw.rsplit("/", 1)[-1][:255] or "video"


def validate_video(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="no filename")
    ext = Path(upload_name(file)).suffix.lower()
    if ext not in settings.allowed_video_extensions:
        raise HTTPException(status_code=400, detail=f"unsupported extension {ext}")
    if file.content_type and not (
        file.content_type.startswith("video/") or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail=f"unsupported content_type {file.content_type}")
    limit = settings.max_video_size_mb * 1024 * 1024
    if file.size is not None and file.size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"file too large: > {settings.max_video_size_mb} MB",
        )


async def save_upload(file: UploadFile, job_id: str) -> str:
    """Stream an upload to the artifact directory with a hard size limit."""
    directory = Path(settings.artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"input_{job_id}{Path(upload_name(file)).suffix.lower()}"
    limit = settings.max_video_size_mb * 1024 * 1024
    written = 0
    completed = False
    try:
        with path.open("wb") as f:
            while chunk := await file.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file too large: > {settings.max_video_size_mb} MB",
                    )
                f.write(chunk)
        completed = True
    finally:
        if not completed:
            remove_artifact(path)
    return str(path)
