"""vibe-result-transforms — Post-processing and result transform pipeline for vibe vision models."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .base import ResultTransform, list_transforms
from .exceptions import TransformError, TransformRequirementError
from .pipeline import TransformPipeline
from .transforms import (
    CharacterIPMapping,
    CleanTags,
    ScoreThresholds,
    TagLevelThresholds,
)

try:
    __version__ = _package_version("vibe-result-transforms")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "CharacterIPMapping",
    "CleanTags",
    "ResultTransform",
    "ScoreThresholds",
    "TagLevelThresholds",
    "TransformError",
    "TransformPipeline",
    "TransformRequirementError",
    "__version__",
    "list_transforms",
]
