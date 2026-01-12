from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "best.onnx")


class Settings(BaseSettings):
    """Application settings."""
    
    # App
    app_name: str = "SoccerGSR ML Service"
    debug: bool = True
    
    # Model
    model_path: str = DEFAULT_MODEL_PATH  # Will be downloaded automatically if missing
    model_gdrive_id: str = "1OA6l1GEb6ki5Dq2zSmJgH4AcEkbYvisq"  # Google Drive file ID
    model_auto_download: bool = True  # Auto-download model if not found
    model_input_size: int = 1280
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./soccer_gsr.db"
    
    # JWT
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Admin token for DELETE /history
    admin_delete_token: str = "admin-delete-token-change-in-production"
    
    # Video processing
    max_video_size_mb: int = 100
    allowed_video_extensions: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    temp_dir: str = "/tmp/soccer_gsr"
    
    # Class names for detection
    class_names: dict[int, str] = {
        0: "player",
        1: "goalkeeper", 
        2: "referee",
        3: "ball"
    }
    
    # Colors for visualization (BGR format for OpenCV)
    class_colors: dict[int, tuple[int, int, int]] = {
        0: (0, 255, 0),      # player - green
        1: (0, 255, 255),    # goalkeeper - yellow
        2: (0, 0, 255),      # referee - red
        3: (0, 165, 255)     # ball - orange
    }
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "protected_namespaces": ("settings_",)
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()

