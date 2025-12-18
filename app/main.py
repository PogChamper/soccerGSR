import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.database import init_db
from app.api import forward, history, stats, auth

import uvicorn


settings = get_settings()


def ensure_model():
    """Ensure model exists, download if necessary."""
    model_path = Path(settings.model_path)
    
    if model_path.exists():
        print(f"Model found at: {model_path}")
        return
    
    if not settings.model_auto_download:
        print(f"WARNING: Model not found at {model_path}")
        print("Set MODEL_AUTO_DOWNLOAD=true or download manually")
        return
    
    print(f"Model not found at {model_path}")
    print("Downloading from Google Drive...")
    
    from app.utils.model_loader import download_model_from_gdrive
    download_model_from_gdrive(
        file_id=settings.model_gdrive_id,
        output_path=model_path
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting {settings.app_name}...")
    
    # Ensure temp directory exists
    os.makedirs(settings.temp_dir, exist_ok=True)
    
    # Ensure model exists (download if needed)
    ensure_model()
    
    # Initialize database
    await init_db()
    print("Database initialized")
    
    # Pre-load detector
    try:
        from app.services.detector import get_detector
        detector = get_detector()
        print(f"Model loaded from: {settings.model_path}")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## SoccerGSR ML Service

ML-сервис для анализа футбольных трансляций.

### Возможности:
- **POST /forward** - Инференс видео через YOLO детектор
- **GET /history** - История запросов
- **DELETE /history** - Удаление истории (требует токен)
- **GET /stats** - Статистика запросов

### Авторизация:
- **POST /auth/register** - Регистрация пользователя
- **POST /auth/login** - Получение JWT токена
- **GET /auth/me** - Информация о текущем пользователе
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forward.router, tags=["inference"])
app.include_router(history.router, tags=["history"])
app.include_router(stats.router, tags=["statistics"])
app.include_router(auth.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint - service info."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "forward": "POST /forward - Run video inference",
            "history": "GET /history - Get request history",
            "delete_history": "DELETE /history - Delete history (requires X-Admin-Token)",
            "stats": "GET /stats - Get statistics",
            "auth": {
                "register": "POST /auth/register - Register user",
                "login": "POST /auth/login - Get JWT token",
                "me": "GET /auth/me - Current user info"
            }
        }
    }


@app.get("/health", tags=["root"])
async def health_check():
    """Health check endpoint."""
    model_status = "unknown"
    try:
        from app.services.detector import get_detector
        detector = get_detector()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "model": model_status
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

