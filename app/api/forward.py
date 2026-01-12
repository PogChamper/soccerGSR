import os
import base64
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import io

from app.config import get_settings
from app.models.database import get_db
from app.models.schemas import ForwardResponse
from app.services.detector import get_detector, YOLODetector
from app.services.video_processor import VideoProcessor
from app.services import history_service

settings = get_settings()
router = APIRouter()


def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file.
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If validation fails
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="bad request")
    
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_video_extensions:
        raise HTTPException(status_code=400, detail="bad request")
    
    # Check content type (allow video/* or application/octet-stream or None)
    if file.content_type:
        allowed_types = ["video/", "application/octet-stream"]
        if not any(file.content_type.startswith(t) for t in allowed_types):
            raise HTTPException(status_code=400, detail="bad request")


async def save_upload_file(file: UploadFile) -> str:
    """Save uploaded file to temp directory.
    
    Args:
        file: Uploaded file
        
    Returns:
        Path to saved file
    """
    os.makedirs(settings.temp_dir, exist_ok=True)
    
    # Create temp file with proper extension
    ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    temp_path = os.path.join(settings.temp_dir, f"upload_{os.urandom(8).hex()}{ext}")
    
    # Write file
    content = await file.read()
    
    # Check file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_video_size_mb:
        raise HTTPException(status_code=400, detail="bad request")
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    return temp_path


@router.post("/forward", response_model=ForwardResponse)
async def forward(
    image: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
    db: AsyncSession = Depends(get_db),
    x_return_format: Optional[str] = Header(default="base64", alias="X-Return-Format")
):
    """Run inference on uploaded video.
    
    Accepts video file via multipart/form-data (parameter: image).
    
    Headers:
        X-Return-Format: "base64" (default) or "stream"
    
    Returns:
        - 200: ForwardResponse with base64 video or StreamingResponse
        - 400: Bad request (invalid format)
        - 403: Model processing error
    """
    input_path = None
    
    try:
        # Validate file
        validate_video_file(image)
        
        # Save uploaded file
        input_path = await save_upload_file(image)
        
        # Initialize detector and processor
        try:
            detector = get_detector()
        except Exception as e:
            # Log error and return 403
            await history_service.create_history_entry(
                db=db,
                status="error",
                error_message=f"Failed to load model: {str(e)}",
                input_filename=image.filename
            )
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )
        
        processor = VideoProcessor(detector)
        
        # Process video
        try:
            video_bytes, metadata, stats = processor.process_video_to_bytes(input_path)
        except Exception as e:
            # Log error and return 403
            await history_service.create_history_entry(
                db=db,
                status="error",
                error_message=f"Processing failed: {str(e)}",
                input_filename=image.filename
            )
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )
        
        # Log success to history
        await history_service.create_history_entry(
            db=db,
            status="success",
            input_filename=metadata.filename,
            input_size_mb=metadata.size_mb,
            input_width=metadata.width,
            input_height=metadata.height,
            input_duration=metadata.duration,
            input_fps=metadata.fps,
            input_frames=metadata.frame_count,
            processing_time=stats.processing_time,
            frames_processed=stats.frames_processed,
            total_detections=stats.total_detections,
            players_count=stats.players_count,
            goalkeepers_count=stats.goalkeepers_count,
            referees_count=stats.referees_count,
            balls_count=stats.balls_count
        )
        
        # Return based on format
        if x_return_format == "stream":
            # Return as streaming response
            return StreamingResponse(
                io.BytesIO(video_bytes),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=processed_{metadata.filename}",
                    "X-Processing-Time": str(stats.processing_time),
                    "X-Detections-Count": str(stats.total_detections),
                    "X-Frames-Processed": str(stats.frames_processed)
                }
            )
        else:
            # Return as base64 in JSON
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
            
            return ForwardResponse(
                status="success",
                video=video_base64,
                metadata={
                    "filename": metadata.filename,
                    "width": metadata.width,
                    "height": metadata.height,
                    "fps": metadata.fps,
                    "duration": metadata.duration,
                    "frame_count": metadata.frame_count,
                    "processing_time": stats.processing_time,
                    "frames_processed": stats.frames_processed,
                    "total_detections": stats.total_detections,
                    "detections_summary": {
                        "players": stats.players_count,
                        "goalkeepers": stats.goalkeepers_count,
                        "referees": stats.referees_count,
                        "balls": stats.balls_count
                    }
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        # Unexpected error - log and return 403
        await history_service.create_history_entry(
            db=db,
            status="error",
            error_message=f"Unexpected error: {str(e)}",
            input_filename=image.filename if image else None
        )
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )
    finally:
        # Cleanup temp file
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

