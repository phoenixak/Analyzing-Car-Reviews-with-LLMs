"""
Logging module for the car reviews analysis project.
"""

import logging
import os
import sys
from typing import Optional

from src.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(
    name: Optional[str] = None, log_file: Optional[str] = None, log_level: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger for the application.

    Args:
        name: Name of the logger. If None, uses the root logger.
        log_file: Path to the log file. If None, uses the default log file.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Determine the logger name
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)

    # Set the logging level
    level = log_level if log_level else LOG_LEVEL
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers to avoid duplicate log messages
    while logger.handlers:
        logger.handlers.pop()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    if log_file is None:
        log_file = LOG_FILE

    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to set up file logging: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience function to get a logger that has already been set up.
    If no logger with that name exists, a new one will be created.

    Args:
        name: Name of the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger doesn't have handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)

    return logger
