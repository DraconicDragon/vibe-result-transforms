"""Character-to-Intellectual Property (Copyright/Series) mapping transform."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, Field
from vibe.metadata import ModelDescriptor, OutputKind
from vibe.results import TagResult
from vibe.session import ModelSession
from vibe.tag_categories import TagCategory

from ..base import ResultTransform
from ..exceptions import TransformRequirementError

logger = logging.getLogger(__name__)

DEFAULT_CHAR_IP_REPO = "SmilingWolf/wd-tagger-character-ip-mappings"
DEFAULT_CHAR_IP_FILE = "character_ip_map.json"


class CharacterIPMappingConfig(BaseModel):
    """Configuration schema for CharacterIPMapping."""

    mapping_file: str | None = Field(
        default=None,
        title="Custom Mapping File",
        description="Path to a custom JSON mapping file. If omitted, uses model bundle or downloads default.",
    )


@dataclass(frozen=True, slots=True)
class CharacterIPMapping(ResultTransform[TagResult]):
    """
    Maps detected character tags to their originating copyright/series IP tags.
    Stores the mapping dictionary in result.extras["character_copyright_mapping"].
    """

    id: ClassVar[str] = "character_ip_mapping"
    display_name: ClassVar[str] = "Character IP Mapping"
    description: ClassVar[str] = "Maps detected character tags to their originating copyright/series IP tags."
    config_model: ClassVar[type[BaseModel] | None] = CharacterIPMappingConfig

    mapping: Mapping[str, Sequence[str]]

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load character mapping from a local JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Character IP mapping file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise TypeError(f"Mapping file '{file_path}' must contain a JSON object.")

        return cls(mapping=data)

    @classmethod
    def from_session(
        cls,
        session: ModelSession,
        *,
        mapping_file: Path | str | None = None,
        hf_repo_id: str = DEFAULT_CHAR_IP_REPO,
        hf_filename: str = DEFAULT_CHAR_IP_FILE,
    ) -> Self:
        """Construct CharacterIPMapping by resolving from file, session artifacts, or Hugging Face."""
        if not cls.is_supported(session):
            raise TransformRequirementError(
                f"Model '{session.model_id}' does not output character tags required for CharacterIPMapping."
            )

        # 1. Explicit file
        if mapping_file is not None:
            return cls.from_file(mapping_file)

        # 2. Check session artifacts for a mapping file
        mapping_artifact = session.artifacts.get_optional("char_mapping") or session.artifacts.get_optional("mapping")
        if mapping_artifact is not None and mapping_artifact.is_file():
            logger.debug("Loading character IP mapping from session artifact: %s", mapping_artifact)
            return cls.from_file(mapping_artifact)

        # 3. Fallback to HF download/cache
        try:
            from huggingface_hub import hf_hub_download

            resolved_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=hf_filename,
                repo_type="model",
            )
            logger.debug("Loaded character IP mapping from HuggingFace (%s): %s", hf_repo_id, resolved_path)
            return cls.from_file(resolved_path)
        except Exception as exc:
            raise TransformRequirementError(
                f"Failed to resolve character IP mapping from '{hf_repo_id}/{hf_filename}': {exc}"
            ) from exc

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        if isinstance(target, ModelSession):
            if not target.is_tagger:
                return False
            categories = [str(c.value if isinstance(c, Enum) else c) for c in target.metadata.output.categories]
            return TagCategory.CHARACTER.value in categories

        if target.output.kind != OutputKind.TAGS:
            return False
        categories = [str(c.value if isinstance(c, Enum) else c) for c in target.output.categories]
        return TagCategory.CHARACTER.value in categories

    def __call__(self, result: TagResult) -> TagResult:
        character_entries = result.category(TagCategory.CHARACTER)
        if not character_entries:
            return result

        detected_mappings: dict[str, list[str]] = {}
        for entry in character_entries:
            ips = self.mapping.get(entry.tag)
            if ips:
                detected_mappings[entry.tag] = list(ips)

        if not detected_mappings:
            return result

        extras = dict(result.extras)
        extras["character_copyright_mapping"] = detected_mappings
        return TagResult(categories=dict(result.categories), extras=extras)
