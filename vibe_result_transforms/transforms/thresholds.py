"""Thresholding transforms: global/category filtering and per-tag optimal thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Self

from pydantic import BaseModel, Field
from vibe.metadata import ModelDescriptor, OutputKind
from vibe.results import TagEntry, TagResult
from vibe.session import ModelSession

from ..base import ResultTransform
from ..exceptions import TransformError, TransformRequirementError

# region UI Schemas


class ScoreThresholdsConfig(BaseModel):
    """Configuration schema for ScoreThresholds."""

    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        title="Base Threshold",
        description="Default minimum score required for tags.",
    )
    category_thresholds: dict[str, float] = Field(
        default_factory=dict,
        title="Category Thresholds",
        description="Per-category threshold overrides (e.g. character: 0.75).",
    )


class TagLevelThresholdsConfig(BaseModel):
    """Configuration schema for TagLevelThresholds."""

    offset: float = Field(
        default=0.0,
        title="Threshold Offset",
        description="Fixed value added to each tag's calibrated threshold.",
    )
    relative_offset: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        title="Relative Offset",
        description="Relative adjustment (-1.0 to 1.0) applied to each threshold.",
    )
    fallback: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Fallback Threshold",
        description="Fallback threshold when a tag has no pre-calibrated value.",
    )


# endregion


@dataclass(frozen=True, slots=True)
class ScoreThresholds(ResultTransform[TagResult]):
    """Filter tags using a base score threshold and optional category overrides."""

    id: ClassVar[str] = "score_thresholds"
    display_name: ClassVar[str] = "Score Thresholds"
    description: ClassVar[str] = "Filter tags using global and category score thresholds."
    config_model: ClassVar[type[BaseModel] | None] = ScoreThresholdsConfig

    threshold: float
    category_thresholds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not (0.0 <= self.threshold <= 1.0):
            raise TransformError(f"threshold must be between 0.0 and 1.0, got {self.threshold}")

        if self.category_thresholds:
            normalized: dict[str, float] = {}
            for cat, val in self.category_thresholds.items():
                cat_key = cat.value if isinstance(cat, Enum) else str(cat)
                if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                    raise TransformError(f"Threshold for category '{cat_key}' must be in [0.0, 1.0], got {val}")
                normalized[cat_key] = float(val)
            object.__setattr__(self, "category_thresholds", normalized)

    @classmethod
    def from_session(
        cls,
        session: ModelSession,
        fallback_threshold: float,
        *,
        category_thresholds: Mapping[str, float] | None = None,
    ) -> Self:
        """
        Construct ScoreThresholds using model recommendations, backed by a mandatory fallback.

        Args:
            session: Active ModelSession.
            fallback_threshold: Mandatory safety floor for any category without a recommended threshold.
            category_thresholds: Explicit category overrides (e.g. {"character": 0.85}).
        """
        if not session.is_tagger:
            raise TransformRequirementError(
                f"ScoreThresholds requires a tagger, but '{session.model_id}' outputs {session.metadata.output.kind}."
            )

        rec = session.tagger.recommendation

        # Base threshold is the model's global recommendation, or your fallback safety floor
        base_threshold = rec.global_threshold if (rec and rec.global_threshold is not None) else fallback_threshold

        # Start with model category recommendations, merge user category overrides
        final_categories = dict(rec.category_thresholds) if rec else {}
        if category_thresholds:
            final_categories.update(category_thresholds)

        return cls(threshold=base_threshold, category_thresholds=final_categories)

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        if isinstance(target, ModelSession):
            return target.is_tagger
        return target.output.kind == OutputKind.TAGS

    def __call__(self, result: TagResult) -> TagResult:
        filtered_categories: dict[str, list[TagEntry]] = {}

        for category, entries in result.categories.items():
            thresh = self.category_thresholds.get(category, self.threshold)
            kept = [e for e in entries if e.score >= thresh]
            if kept:
                filtered_categories[category] = kept

        return TagResult(categories=filtered_categories, extras=dict(result.extras))


@dataclass(frozen=True, slots=True)
class TagLevelThresholds(ResultTransform[TagResult]):
    """Filter tags using individual calibrated per-tag decision thresholds."""

    id: ClassVar[str] = "tag_level_thresholds"
    display_name: ClassVar[str] = "Tag Level Thresholds"
    description: ClassVar[str] = "Filter tags using model-provided per-tag calibrated decision thresholds."
    config_model: ClassVar[type[BaseModel] | None] = TagLevelThresholdsConfig

    threshold_map: Mapping[str, float]
    offset: float = 0.0
    relative_offset: float = 0.0
    fallback: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.offset):
            raise TransformError(f"offset must be a finite number, got {self.offset}")
        if not math.isfinite(self.relative_offset) or not (-1.0 <= self.relative_offset <= 1.0):
            raise TransformError(f"relative_offset must be in [-1.0, 1.0], got {self.relative_offset}")
        if self.fallback is not None and (not math.isfinite(self.fallback) or not (0.0 <= self.fallback <= 1.0)):
            raise TransformError(f"fallback must be in [0.0, 1.0], got {self.fallback}")

    @classmethod
    def from_session(
        cls,
        session: ModelSession,
        *,
        offset: float = 0.0,
        relative_offset: float = 0.0,
        fallback: float | None = None,
    ) -> Self:
        """Construct TagLevelThresholds directly from the session's ThresholdTable."""
        if not session.is_tagger:
            raise TransformRequirementError(
                f"TagLevelThresholds requires a tagger, but '{session.model_id}' outputs {session.metadata.output.kind}."
            )

        if session.tagger.thresholds is None:
            raise TransformRequirementError(
                f"Model '{session.model_id}' does not provide per-tag calibrated thresholds (ThresholdProvider). "
                f"Use ScoreThresholds instead."
            )

        return cls(
            threshold_map=session.tagger.thresholds.values,
            offset=offset,
            relative_offset=relative_offset,
            fallback=fallback,
        )

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        if isinstance(target, ModelSession):
            return target.is_tagger and target.tagger.thresholds is not None
        return "ThresholdProvider" in target.capabilities

    def __call__(self, result: TagResult) -> TagResult:
        filtered_categories: dict[str, list[TagEntry]] = {}

        for category, entries in result.categories.items():
            kept: list[TagEntry] = []
            for entry in entries:
                thresh = self.threshold_map.get(entry.tag, self.fallback)
                if thresh is not None:
                    thresh += self.offset
                    if self.relative_offset != 0.0:
                        thresh *= 1.0 + self.relative_offset

                if thresh is None or entry.score >= thresh:
                    kept.append(entry)

            if kept:
                filtered_categories[category] = kept

        return TagResult(categories=filtered_categories, extras=dict(result.extras))
