from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_db
from app.models.schemas import HistoryResponse, RequestHistoryItem, DeleteHistoryResponse
from app.services import history_service

settings = get_settings()
router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, regex="^(success|error)$")
):
    """Get request history.
    
    Query Parameters:
        limit: Maximum number of entries (default: 100, max: 1000)
        offset: Offset for pagination (default: 0)
        status: Filter by status ('success' or 'error')
    
    Returns:
        HistoryResponse with list of history items
    """
    entries, total = await history_service.get_history(
        db=db,
        limit=limit,
        offset=offset,
        status_filter=status
    )
    
    items = [RequestHistoryItem.model_validate(entry) for entry in entries]
    
    return HistoryResponse(total=total, items=items)


@router.delete("/history", response_model=DeleteHistoryResponse)
async def delete_history(
    db: AsyncSession = Depends(get_db),
    x_admin_token: str = Header(..., alias="X-Admin-Token")
):
    """Delete all request history.
    
    Requires admin token in X-Admin-Token header.
    
    Headers:
        X-Admin-Token: Admin verification token (required)
    
    Returns:
        DeleteHistoryResponse with count of deleted entries
    
    Raises:
        401: Invalid or missing token
    """
    # Verify admin token
    if x_admin_token != settings.admin_delete_token:
        raise HTTPException(
            status_code=401,
            detail="unauthorized"
        )
    
    deleted_count = await history_service.delete_all_history(db)
    
    return DeleteHistoryResponse(
        deleted_count=deleted_count,
        message=f"Successfully deleted {deleted_count} history entries"
    )

