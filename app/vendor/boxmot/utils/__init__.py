# Vendored from BoxMOT (AGPL-3.0). Trimmed: only the things our BotSort needs.

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (re-exported for downstream convenience)
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models"  # placeholder; we never download via boxmot

NUM_THREADS = min(8, max(1, (os.cpu_count() or 2) - 1))


def _is_main_process(record):
    return record["process"].name == "MainProcess"


def configure_logging(main_only: bool = True):
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        filter=_is_main_process if main_only else None,
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )
    return logger


configure_logging()
