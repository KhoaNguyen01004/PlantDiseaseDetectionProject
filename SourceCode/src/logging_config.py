"""
Centralized logging configuration for PlantDiseaseDetection.

This module provides a unified logger factory that ensures consistent
logging behavior across training, inference, validation, and export pipelines.

Usage:
    from src.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Starting training...")
    logger.warning("Batch skipped due to quality check")
    logger.error("Failed to load model", exc_info=True)

Configuration:
    - Log level: Read from config.yaml (default: INFO)
    - Format: [TIMESTAMP] [LEVEL] [MODULE] - MESSAGE
    - Handlers: Console (all levels) + File (INFO+)
    - File rotation: Automatic daily rotation
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_log_level_from_config() -> str:
    """Load log level from config.yaml if available."""
    try:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
                return config.get("logging", {}).get("level", "INFO").upper()
    except Exception:
        pass
    return "INFO"


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG/INFO/WARNING/ERROR). If None, read from config.
        log_dir: Directory for log files. If None, use SourceCode/logs/
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Skip reconfiguration if logger already has handlers
    if logger.handlers:
        return logger
    
    # Determine log level
    if level is None:
        level = load_log_level_from_config()
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate logs from parent loggers
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        fmt="[%(levelname)s] %(name)s - %(message)s"
    )
    
    # Console handler (always INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to create file handler: {e}")
    
    return logger


def disable_external_logging() -> None:
    """
    Suppress verbose logging from third-party libraries.
    
    Call this early in main execution to reduce noise from PyTorch,
    TensorFlow, Pillow, etc.
    """
    for lib_name in ["torch", "tensorflow", "PIL", "urllib3", "torchvision"]:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)


# Module-level logger for this file
_logger = get_logger(__name__)
