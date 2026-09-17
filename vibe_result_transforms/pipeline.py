"""Sequential pipeline for applying multiple result transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, overload

from vibe.results import InferenceResult, InferenceResultItem, ModelResult, TagResult

from .base import ResultTransform


@dataclass(frozen=True, slots=True)
class TransformPipeline:
    """
    Executes an ordered sequence of transforms on a TagResult or InferenceResult envelope.

    Example:
        pipeline = TransformPipeline([
            ScoreThresholds.from_session(session),
            CharacterIPMapping.from_session(session),
            CleanTags(),
        ])

        # Works on single results:
        filtered = pipeline(result)

        # Works on batch envelopes:
        batch_filtered = pipeline(session.infer(["img1.png", "img2.png"]))
    """

    transforms: tuple[ResultTransform[Any], ...]

    def __init__(self, transforms: Sequence[ResultTransform[Any]]) -> None:
        object.__setattr__(self, "transforms", tuple(transforms))

    @overload
    def __call__(self, target: InferenceResult) -> InferenceResult: ...

    @overload
    def __call__(self, target: TagResult) -> TagResult: ...

    @overload
    def __call__(self, target: ModelResult) -> ModelResult: ...

    def __call__(self, target: Any) -> Any:
        # Batch envelope transparent handling
        if isinstance(target, InferenceResult):
            transformed_items = [
                InferenceResultItem(
                    index=item.index,
                    input_ref=item.input_ref,
                    result=self._apply_single(item.result),
                )
                for item in target.items
            ]
            return InferenceResult(
                total_inputs=target.total_inputs,
                items=transformed_items,
                memory=target.memory,
            )

        # Single result
        return self._apply_single(target)

    def _apply_single(self, result: Any) -> Any:
        curr = result
        for tf in self.transforms:
            curr = tf(curr)
        return curr
