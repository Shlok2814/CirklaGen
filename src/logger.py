"""Logging system for CirklaGen.

Ensures that logs are written to both the terminal console and to outputs/logs/cirklagen.log.
No silent failures.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logger(
    name: str = "cirklagen",
    log_file: str | Path = "outputs/logs/cirklagen.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger that logs to both console and a log file.

    Args:
        name: Name of the logger.
        log_file: Path to the log file.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already initialized
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger if already handling
    logger.propagate = False

    return logger


# Default application logger instance
logger = setup_logger()


if __name__ == "__main__":
    test_logger = setup_logger("cirklagen_demo")
    print("--- Running logger demo ---")
    test_logger.info("This is an INFO message for testing.")
    test_logger.warning("This is a WARNING message for testing.")
    test_logger.error("This is an ERROR message for testing.")
    print(f"Log written to outputs/logs/cirklagen.log (Verified: {os.path.exists('outputs/logs/cirklagen.log')})")
