"""Exceptions for result transforms and pipeline execution."""

from __future__ import annotations


class TransformError(Exception):
    """Base exception for all transform errors."""


class TransformRequirementError(TransformError):
    """Raised when a transform requires model data that the session does not provide."""
