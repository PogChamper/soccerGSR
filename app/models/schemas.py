from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============ Detection Schemas ============

class Detection(BaseModel):
    """Single detection result."""
    
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(description="[x1, y1, x2, y2]")


class FrameDetections(BaseModel):
    """Detections for a single frame."""
    
    frame_index: int
    detections: List[Detection]


# ============ Forward Request/Response ============

class ForwardResponse(BaseModel):
    """Response for /forward endpoint with video output."""
    
    status: str
    video: Optional[str] = Field(None, description="Base64 encoded video")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForwardErrorResponse(BaseModel):
    """Error response."""
    
    detail: str


# ============ History Schemas ============

class RequestHistoryItem(BaseModel):
    """Single history item."""
    
    id: int
    request_id: str
    timestamp: datetime
    status: str
    error_message: Optional[str] = None
    
    # Input
    input_filename: Optional[str] = None
    input_size_mb: Optional[float] = None
    input_width: Optional[int] = None
    input_height: Optional[int] = None
    input_duration: Optional[float] = None
    input_fps: Optional[float] = None
    input_frames: Optional[int] = None
    
    # Processing
    processing_time: Optional[float] = None
    frames_processed: Optional[int] = None
    
    # Detections
    total_detections: Optional[int] = None
    players_count: Optional[int] = None
    goalkeepers_count: Optional[int] = None
    referees_count: Optional[int] = None
    balls_count: Optional[int] = None
    
    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """Response for /history endpoint."""
    
    total: int
    items: List[RequestHistoryItem]


class DeleteHistoryResponse(BaseModel):
    """Response for DELETE /history."""
    
    deleted_count: int
    message: str


# ============ Stats Schemas ============

class PercentileStats(BaseModel):
    """Percentile statistics."""
    
    mean: float
    p50: float
    p95: float
    p99: float


class ResolutionDistribution(BaseModel):
    """Resolution distribution."""
    
    most_common: Optional[str] = None
    distribution: Dict[str, int] = Field(default_factory=dict)


class InputCharacteristics(BaseModel):
    """Input video characteristics statistics."""
    
    video_duration: Dict[str, float] = Field(default_factory=dict)
    resolution: ResolutionDistribution = Field(default_factory=ResolutionDistribution)
    file_size_mb: Dict[str, float] = Field(default_factory=dict)


class DetectionStats(BaseModel):
    """Detection statistics."""
    
    avg_per_frame: float = 0.0
    by_class: Dict[str, float] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Response for /stats endpoint."""
    
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    processing_time: PercentileStats
    input_characteristics: InputCharacteristics
    detections: DetectionStats


# ============ Auth Schemas (PRO) ============

class UserCreate(BaseModel):
    """User registration schema."""
    
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6)


class UserResponse(BaseModel):
    """User response schema."""
    
    id: int
    username: str
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response."""
    
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    
    username: Optional[str] = None
    user_id: Optional[int] = None
    is_admin: bool = False

