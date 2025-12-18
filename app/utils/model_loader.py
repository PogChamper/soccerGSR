import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Google Drive file ID from the sharing link
# https://drive.google.com/file/d/1OA6l1GEb6ki5Dq2zSmJgH4AcEkbYvisq/view?usp=sharing
GDRIVE_FILE_ID = "1OA6l1GEb6ki5Dq2zSmJgH4AcEkbYvisq"
MODEL_FILENAME = "best.onnx"

# Default model directory (inside project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / MODEL_FILENAME


def get_model_path() -> Path:
    """Get the path to the model file.
    
    Returns:
        Path to the model file
    """
    # Check environment variable first
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return Path(env_path)
    
    return DEFAULT_MODEL_PATH


def download_model_from_gdrive(
    file_id: str = GDRIVE_FILE_ID,
    output_path: Optional[Path] = None,
    quiet: bool = False
) -> Path:
    """Download model from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Where to save the model (default: models/best.onnx)
        quiet: Suppress download progress
        
    Returns:
        Path to downloaded model
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading from Google Drive. "
            "Install it with: pip install gdown"
        )
    
    if output_path is None:
        output_path = DEFAULT_MODEL_PATH
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build Google Drive URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    logger.info(f"Downloading model from Google Drive...")
    logger.info(f"File ID: {file_id}")
    logger.info(f"Output path: {output_path}")
    
    # Download file
    gdown.download(url, str(output_path), quiet=quiet)
    
    # Verify download
    if not output_path.exists():
        raise RuntimeError(f"Download failed: {output_path} does not exist")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model downloaded successfully: {file_size_mb:.1f} MB")
    
    return output_path


def ensure_model_exists(
    model_path: Optional[Path] = None,
    auto_download: bool = True
) -> Path:
    """Ensure the model file exists, downloading if necessary.
    
    Args:
        model_path: Path to check (default: from get_model_path())
        auto_download: Whether to automatically download if missing
        
    Returns:
        Path to the model file
        
    Raises:
        FileNotFoundError: If model doesn't exist and auto_download is False
    """
    if model_path is None:
        model_path = get_model_path()
    
    if model_path.exists():
        logger.info(f"Model found at: {model_path}")
        return model_path
    
    if not auto_download:
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Download it manually or set auto_download=True"
        )
    
    logger.info(f"Model not found at {model_path}, downloading...")
    return download_model_from_gdrive(output_path=model_path)


def get_model_info(model_path: Optional[Path] = None) -> dict:
    """Get information about the model file.
    
    Args:
        model_path: Path to model (default: from get_model_path())
        
    Returns:
        Dictionary with model info
    """
    if model_path is None:
        model_path = get_model_path()
    
    if not model_path.exists():
        return {
            "exists": False,
            "path": str(model_path),
            "size_mb": None
        }
    
    return {
        "exists": True,
        "path": str(model_path),
        "size_mb": model_path.stat().st_size / (1024 * 1024)
    }


if __name__ == "__main__":
    # CLI for manual download
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Download SoccerGSR model")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    if args.output.exists() and not args.force:
        print(f"Model already exists at {args.output}")
        print("Use --force to re-download")
    else:
        download_model_from_gdrive(output_path=args.output)
        print(f"Model saved to {args.output}")

