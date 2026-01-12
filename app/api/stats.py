from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import (
    StatsResponse, 
    PercentileStats, 
    InputCharacteristics,
    ResolutionDistribution,
    DetectionStats
)
from app.services import history_service

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregated statistics for all requests.
    
    Returns:
        StatsResponse with:
        - Total/successful/failed request counts
        - Processing time percentiles (mean, p50, p95, p99)
        - Input characteristics (duration, resolution, file size)
        - Detection statistics (avg per frame, by class)
    """
    stats_data = await history_service.get_stats(db)
    
    # Build response with proper structure
    processing_time = PercentileStats(
        mean=stats_data["processing_time"]["mean"],
        p50=stats_data["processing_time"]["p50"],
        p95=stats_data["processing_time"]["p95"],
        p99=stats_data["processing_time"]["p99"]
    )
    
    resolution = ResolutionDistribution(
        most_common=stats_data["input_characteristics"]["resolution"]["most_common"],
        distribution=stats_data["input_characteristics"]["resolution"]["distribution"]
    )
    
    input_characteristics = InputCharacteristics(
        video_duration=stats_data["input_characteristics"]["video_duration"],
        resolution=resolution,
        file_size_mb=stats_data["input_characteristics"]["file_size_mb"]
    )
    
    detections = DetectionStats(
        avg_per_frame=stats_data["detections"]["avg_per_frame"],
        by_class=stats_data["detections"]["by_class"]
    )
    
    return StatsResponse(
        total_requests=stats_data["total_requests"],
        successful_requests=stats_data["successful_requests"],
        failed_requests=stats_data["failed_requests"],
        processing_time=processing_time,
        input_characteristics=input_characteristics,
        detections=detections
    )

