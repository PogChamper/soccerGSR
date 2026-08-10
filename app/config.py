from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings.

    Model artifact paths and sources are not configured here; they live in
    app.utils.models_registry. Use the MODELS__{NAME}__PATH env override for
    ad-hoc model paths.
    """

    # App
    app_name: str = "SoccerGSR ML Service"
    debug: bool = False

    # Auto-download registered models on startup (see models_registry.REGISTRY)
    model_auto_download: bool = True

    # DEIMv2 detector (DETR-style, postproc baked into the graph, no NMS).
    # Trained with num_classes=5 where index 0 is background; service cls_id =
    # model_label - 1 (0=player, 1=goalkeeper, 2=referee, 3=ball).
    deimv2_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Database
    database_url: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'soccer_gsr.db'}"

    # Video processing
    max_video_size_mb: int = Field(default=100, gt=0)
    max_pending_jobs: int = Field(default=4, ge=1)
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    artifact_dir: str = str(PROJECT_ROOT / "soccer_gsr_jobs")

    # Class names for detection
    class_names: dict[int, str] = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

    # Colors for visualization (BGR format for OpenCV)
    class_colors: dict[int, tuple[int, int, int]] = {
        0: (0, 255, 0),  # player - green
        1: (0, 255, 255),  # goalkeeper - yellow
        2: (0, 0, 255),  # referee - red
        3: (0, 165, 255),  # ball - orange
    }

    model_config = {"env_file": ".env", "extra": "ignore", "protected_namespaces": ("settings_",)}


@lru_cache
def get_settings() -> Settings:
    return Settings()
