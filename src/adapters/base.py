"""Base exceptions and shared utilities for CirklaGen model adapters."""

from __future__ import annotations


class NotConfigured(Exception):
    """Raised when required API credentials for a model backend are missing in .env."""
    pass


NotConfiguredError = NotConfigured
