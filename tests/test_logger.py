"""Unit tests for logger module."""

import logging
from pathlib import Path
from src.logger import setup_logger


def test_logger_writes_to_file(tmp_path: Path):
    """Verify that logger writes records to specified file and console."""
    test_log_file = tmp_path / "test_logs" / "test.log"
    logger = setup_logger(name="test_logger", log_file=test_log_file, level=logging.DEBUG)

    test_message = "Test log entry for verification"
    logger.info(test_message)

    # Flush handlers to ensure file output
    for handler in logger.handlers:
        handler.flush()

    assert test_log_file.exists(), "Log file was not created"
    content = test_log_file.read_text(encoding="utf-8")
    assert test_message in content
