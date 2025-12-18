from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
import numpy as np

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import RequestHistory


async def create_history_entry(
    db: AsyncSession,
    status: str,
    error_message: Optional[str] = None,
    input_filename: Optional[str] = None,
    input_size_mb: Optional[float] = None,
    input_width: Optional[int] = None,
    input_height: Optional[int] = None,
    input_duration: Optional[float] = None,
    input_fps: Optional[float] = None,
    input_frames: Optional[int] = None,
    processing_time: Optional[float] = None,
    frames_processed: Optional[int] = None,
    total_detections: Optional[int] = None,
    players_count: Optional[int] = None,
    goalkeepers_count: Optional[int] = None,
    referees_count: Optional[int] = None,
    balls_count: Optional[int] = None,
    user_id: Optional[int] = None
) -> RequestHistory:
    """Create a new history entry.
    
    Args:
        db: Database session
        status: Request status ('success' or 'error')
        ... other parameters
        
    Returns:
        Created RequestHistory object
    """
    entry = RequestHistory(
        request_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        status=status,
        error_message=error_message,
        input_filename=input_filename,
        input_size_mb=input_size_mb,
        input_width=input_width,
        input_height=input_height,
        input_duration=input_duration,
        input_fps=input_fps,
        input_frames=input_frames,
        processing_time=processing_time,
        frames_processed=frames_processed,
        total_detections=total_detections,
        players_count=players_count,
        goalkeepers_count=goalkeepers_count,
        referees_count=referees_count,
        balls_count=balls_count,
        user_id=user_id
    )
    
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    
    return entry


async def get_history(
    db: AsyncSession,
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None
) -> tuple[List[RequestHistory], int]:
    """Get history entries with pagination.
    
    Args:
        db: Database session
        limit: Maximum entries to return
        offset: Offset for pagination
        status_filter: Optional status filter
        
    Returns:
        Tuple of (history entries, total count)
    """
    # Build query
    query = select(RequestHistory).order_by(RequestHistory.timestamp.desc())
    count_query = select(func.count(RequestHistory.id))
    
    if status_filter:
        query = query.where(RequestHistory.status == status_filter)
        count_query = count_query.where(RequestHistory.status == status_filter)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get paginated results
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    entries = result.scalars().all()
    
    return list(entries), total


async def delete_all_history(db: AsyncSession) -> int:
    """Delete all history entries.
    
    Args:
        db: Database session
        
    Returns:
        Number of deleted entries
    """
    # Get count first
    count_result = await db.execute(select(func.count(RequestHistory.id)))
    count = count_result.scalar()
    
    # Delete all
    await db.execute(delete(RequestHistory))
    await db.commit()
    
    return count


async def get_stats(db: AsyncSession) -> Dict[str, Any]:
    """Calculate statistics from history.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with statistics
    """
    # Get all successful entries for stats
    result = await db.execute(
        select(RequestHistory).where(RequestHistory.status == "success")
    )
    successful_entries = result.scalars().all()
    
    # Get counts
    total_result = await db.execute(select(func.count(RequestHistory.id)))
    total_requests = total_result.scalar()
    
    success_result = await db.execute(
        select(func.count(RequestHistory.id)).where(RequestHistory.status == "success")
    )
    successful_requests = success_result.scalar()
    
    failed_requests = total_requests - successful_requests
    
    # Calculate processing time stats
    processing_times = [e.processing_time for e in successful_entries if e.processing_time is not None]
    
    if processing_times:
        processing_time_stats = {
            "mean": float(np.mean(processing_times)),
            "p50": float(np.percentile(processing_times, 50)),
            "p95": float(np.percentile(processing_times, 95)),
            "p99": float(np.percentile(processing_times, 99))
        }
    else:
        processing_time_stats = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    
    # Calculate input characteristics
    durations = [e.input_duration for e in successful_entries if e.input_duration is not None]
    sizes = [e.input_size_mb for e in successful_entries if e.input_size_mb is not None]
    
    duration_stats = {}
    if durations:
        duration_stats = {
            "mean": float(np.mean(durations)),
            "min": float(min(durations)),
            "max": float(max(durations))
        }
    
    size_stats = {}
    if sizes:
        size_stats = {
            "mean": float(np.mean(sizes)),
            "p50": float(np.percentile(sizes, 50)),
            "p95": float(np.percentile(sizes, 95))
        }
    
    # Resolution distribution
    resolution_counts = {}
    for e in successful_entries:
        if e.input_width and e.input_height:
            res = f"{e.input_width}x{e.input_height}"
            resolution_counts[res] = resolution_counts.get(res, 0) + 1
    
    most_common_resolution = None
    if resolution_counts:
        most_common_resolution = max(resolution_counts, key=resolution_counts.get)
    
    # Detection stats
    total_detections = sum(e.total_detections or 0 for e in successful_entries)
    total_frames = sum(e.frames_processed or 0 for e in successful_entries)
    avg_per_frame = total_detections / total_frames if total_frames > 0 else 0.0
    
    players_total = sum(e.players_count or 0 for e in successful_entries)
    goalkeepers_total = sum(e.goalkeepers_count or 0 for e in successful_entries)
    referees_total = sum(e.referees_count or 0 for e in successful_entries)
    balls_total = sum(e.balls_count or 0 for e in successful_entries)
    
    # Average per frame by class
    detection_by_class = {}
    if total_frames > 0:
        detection_by_class = {
            "player": players_total / total_frames,
            "goalkeeper": goalkeepers_total / total_frames,
            "referee": referees_total / total_frames,
            "ball": balls_total / total_frames
        }
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "processing_time": processing_time_stats,
        "input_characteristics": {
            "video_duration": duration_stats,
            "resolution": {
                "most_common": most_common_resolution,
                "distribution": resolution_counts
            },
            "file_size_mb": size_stats
        },
        "detections": {
            "avg_per_frame": avg_per_frame,
            "by_class": detection_by_class
        }
    }

