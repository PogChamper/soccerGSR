from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import StatsResponse
from app.services import history_service

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Aggregated request statistics: counts, processing-time percentiles,
    input characteristics, detections per frame by class."""
    return StatsResponse(**await history_service.get_stats(db))
