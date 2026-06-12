from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings.

    Model artifact paths/sources are NOT configured here — they live in
    ``app.utils.models_registry`` (single source of truth). Use the
    ``MODELS__{NAME}__PATH`` env override for ad-hoc model paths.
    """
    
    # App
    app_name: str = "SoccerGSR ML Service"
    debug: bool = True
    
    # Detector backend selection: "deimv2" (DEIMv2-DINOv3 M@896, DETR-style) or
    # "yolo" (legacy YOLOv5lu@1280). Override via env DETECTOR_BACKEND.
    detector_backend: str = "deimv2"

    # Auto-download registered models on startup (see models_registry.REGISTRY)
    model_auto_download: bool = True

    # YOLO detector (legacy backend)
    model_input_size: int = 1280
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # DEIMv2 detector (DETR-style, postproc baked into the graph, no NMS).
    # Trained with num_classes=5 where index 0 is background; service cls_id =
    # model_label - 1 (0=player, 1=goalkeeper, 2=referee, 3=ball).
    deimv2_input_size: int = 896
    deimv2_confidence_threshold: float = 0.4
    deimv2_model_size: str = "m"  # affects normalization (s/m/l/x use ImageNet)
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./soccer_gsr.db"
    
    # JWT
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
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

