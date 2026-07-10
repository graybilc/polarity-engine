# src/config.py
import logging.config
import os
from pathlib import Path

# =====================================================================
# 1. PATH CONFIGURATION
# =====================================================================
# Automatically locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure runtime directories exist
LOG_DIR.mkdir(exist_ok=True)

# =====================================================================
# 2. LOGGING CONFIGURATION MATRIX
# =====================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    # Formatters: Define the layout of your log strings
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    # Handlers: Define where the logs actually go
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "WARNING",  # Captures WARNING, ERROR, and CRITICAL
            "formatter": "standard",
            "filename": LOG_DIR / "pipeline.log",
            "maxBytes": 1048576,  # 1 MB per file before rotating
            "backupCount": 3,
            "encoding": "utf8",
        },
    },
    # Root Logger: The master configuration inherited by all files
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",  # Global capture threshold
    },
}

# =====================================================================
# 3. INITIALIZATION FUNCTION
# =====================================================================
def setup_logging():
    """Initializes the unified logging infrastructure for the pipeline."""
    logging.config.dictConfig(LOGGING_CONFIG)