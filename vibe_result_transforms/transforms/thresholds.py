"""Thresholding transforms: global/category filtering and per-tag optimal thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Self

import numpy as np
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
        # 1. FAST PATH: Filter directly from NumPy arrays without materializing 30,000 throwaway TagEntries
        if (
            result._categories is None
            and result._scores is not None
            and result._tag_names is not None
            and result._category_indices is not None
        ):
            scores = result._scores
            names = result._tag_names
            usable_count = min(len(scores), len(names))
            filtered_categories: dict[str, list[TagEntry]] = {}

            for cat_name, indices in result._category_indices.items():
                thresh = self.category_thresholds.get(cat_name, self.threshold)
                valid_indices = [idx for idx in indices if idx < usable_count]
                if not valid_indices:
                    continue

                idx_arr = np.array(valid_indices, dtype=np.int32)
                cat_scores = scores[idx_arr]

                # Fast C-level boolean mask: only keep indices above threshold!
                mask = cat_scores >= thresh
                if not np.any(mask):
                    continue

                passed_idx = idx_arr[mask]
                passed_scores = cat_scores[mask]

                # Sort descending
                sort_order = np.argsort(-passed_scores)
                sorted_idx = passed_idx[sort_order]
                sorted_scores = passed_scores[sort_order]

                # ONLY instantiate TagEntry for the surviving tags (~50 objects instead of 30,000!)
                filtered_categories[cat_name] = [
                    TagEntry(tag=names[i], score=float(s)) for i, s in zip(sorted_idx, sorted_scores, strict=False)
                ]

            return TagResult(categories=filtered_categories, extras=dict(result.extras))

        # 2. FALLBACK PATH: If result was already materialized
        filtered_categories = {}
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
        # 1. FAST PATH: Filter directly from NumPy arrays without materializing 30,000 throwaway TagEntries
        if (
            result._categories is None
            and result._scores is not None
            and result._tag_names is not None
            and result._category_indices is not None
        ):
            scores = result._scores
            names = result._tag_names
            usable_count = min(len(scores), len(names))
            filtered_categories: dict[str, list[TagEntry]] = {}

            offset = self.offset
            rel_offset = self.relative_offset
            has_rel = rel_offset != 0.0
            thresh_map = self.threshold_map
            fallback = self.fallback

            for cat_name, indices in result._category_indices.items():
                valid_indices = [idx for idx in indices if idx < usable_count]
                if not valid_indices:
                    continue

                idx_arr = np.array(valid_indices, dtype=np.int32)
                cat_scores = scores[idx_arr]

                # Sort by score descending first
                sort_order = np.argsort(-cat_scores)
                sorted_idx = idx_arr[sort_order]
                sorted_scores = cat_scores[sort_order]

                kept: list[TagEntry] = []
                for i, s in zip(sorted_idx, sorted_scores, strict=False):
                    tag_name = names[i]
                    score_val = float(s)

                    thresh = thresh_map.get(tag_name, fallback)
                    if thresh is not None:
                        thresh += offset
                        if has_rel:
                            thresh *= 1.0 + rel_offset

                    # If score passes threshold, ONLY THEN instantiate TagEntry!
                    if thresh is None or score_val >= thresh:
                        kept.append(TagEntry(tag=tag_name, score=score_val))

                if kept:
                    filtered_categories[cat_name] = kept

            return TagResult(categories=filtered_categories, extras=dict(result.extras))

        # 2. FALLBACK PATH: If result was already materialized
        filtered_categories = {}
        for category, entries in result.categories.items():
            kept = []
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
