from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, ForeignKey, create_engine
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship

from app.config import get_settings

Base = declarative_base()
settings = get_settings()


class User(Base):
    """User model for JWT authentication (PRO)."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to request history
    requests = relationship("RequestHistory", back_populates="user")


class RequestHistory(Base):
    """Request history model."""
    
    __tablename__ = "request_history"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(20), nullable=False)  # success, error
    error_message = Column(Text, nullable=True)
    
    # Input characteristics
    input_filename = Column(String(255), nullable=True)
    input_size_mb = Column(Float, nullable=True)
    input_width = Column(Integer, nullable=True)
    input_height = Column(Integer, nullable=True)
    input_duration = Column(Float, nullable=True)  # seconds
    input_fps = Column(Float, nullable=True)
    input_frames = Column(Integer, nullable=True)
    
    # Processing stats
    processing_time = Column(Float, nullable=True)  # seconds
    frames_processed = Column(Integer, nullable=True)
    
    # Detection results (aggregates)
    total_detections = Column(Integer, nullable=True)
    players_count = Column(Integer, nullable=True)
    goalkeepers_count = Column(Integer, nullable=True)
    referees_count = Column(Integer, nullable=True)
    balls_count = Column(Integer, nullable=True)
    
    # User reference
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="requests")


# Async engine and session
engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

