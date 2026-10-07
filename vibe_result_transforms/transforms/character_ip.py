"""Character-to-Intellectual Property (Copyright/Series) mapping transform."""

from __future__ import annotations

import ast
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

DEFAULT_CHAR_IP_REPO = "deepghs/pixai-tagger-v0.9-onnx"
DEFAULT_CHAR_IP_FILE = "selected_tags.csv"

_DEFAULT_MAPPING_CACHE: dict[str, list[str]] | None = None


# region File Parsers


def _safe_parse_list(value: object) -> list[str]:
    """Parse a serialized list of IP strings from JSON, escaped JSON, or Python literals."""
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass

    if '\\"' in text:
        try:
            parsed = json.loads(text.replace('\\"', '"'))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    return []


def _load_mapping_csv(path: Path) -> dict[str, list[str]]:
    """Parse CSV mapping files containing 'name' and 'ips' columns."""
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue

            ips = _safe_parse_list(row.get("ips", "[]"))
            if ips:
                # Canonicalize key with underscores
                out[name.replace(" ", "_")] = ips

    return out


def _load_mapping_json(path: Path) -> dict[str, list[str]]:
    """Parse JSON mapping files supporting both flat mappings and tag_map/ips_by_tag_id schemas."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Failed to load character mapping JSON %s: %s", path, exc)
        return {}

    if not isinstance(raw, dict):
        return {}

    # Format A: {"char_name": ["ip1", "ip2"]} or {"mapping": {"char_name": [...]}}
    candidate = raw.get("mapping") if isinstance(raw.get("mapping"), dict) else raw
    out: dict[str, list[str]] = {}
    for key, value in candidate.items():
        if not key or not isinstance(value, (list, tuple, set)) or not value:
            continue
        out[str(key).replace(" ", "_")] = [str(x) for x in value]

    if out:
        return out

    # Format B: {"tag_map": {"char": 123}, "ips_by_tag_id": {"123": [...]}}
    tag_map = raw.get("tag_map")
    ips_by_tag_id = raw.get("ips_by_tag_id")
    if isinstance(tag_map, dict) and isinstance(ips_by_tag_id, dict):
        for tag_name, tag_id in tag_map.items():
            ips = ips_by_tag_id.get(str(tag_id), ips_by_tag_id.get(tag_id))
            if isinstance(ips, (list, tuple, set)) and ips:
                out[str(tag_name).replace(" ", "_")] = [str(x) for x in ips]

    return out


def load_mapping_file(path: Path | str) -> dict[str, list[str]]:
    """Load character-to-IP mapping from either CSV or JSON format."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Character IP mapping file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _load_mapping_csv(file_path)
    if suffix in {".json", ".js"}:
        return _load_mapping_json(file_path)

    raise ValueError(f"Unsupported character IP mapping format: '{file_path.name}' (expected .csv or .json)")


def _resolve_default_mapping(
    hf_repo_id: str = DEFAULT_CHAR_IP_REPO,
    hf_filename: str = DEFAULT_CHAR_IP_FILE,
) -> dict[str, list[str]]:
    """Download or read default mapping from Hugging Face hub with in-memory caching [1]."""
    global _DEFAULT_MAPPING_CACHE
    if _DEFAULT_MAPPING_CACHE is not None:
        return _DEFAULT_MAPPING_CACHE

    try:
        from huggingface_hub import hf_hub_download

        resolved_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type="model",
        )
        logger.debug("Resolved default character IP mapping from HuggingFace (%s): %s", hf_repo_id, resolved_path)
        _DEFAULT_MAPPING_CACHE = load_mapping_file(resolved_path)
        return _DEFAULT_MAPPING_CACHE
    except Exception as exc:
        raise TransformRequirementError(
            f"Failed to resolve default character IP mapping from '{hf_repo_id}/{hf_filename}': {exc}"
        ) from exc


# endregion


class CharacterIPMappingConfig(BaseModel):
    """Configuration schema for CharacterIPMapping."""

    mapping_file: str | None = Field(
        default=None,
        title="Custom Mapping File",
        description="Path to a custom CSV or JSON mapping file. If omitted, downloads default community mapping.",
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

    mapping: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # If no mapping was explicitly supplied, resolve the default community mapping
        if not self.mapping:
            resolved = _resolve_default_mapping()
            object.__setattr__(self, "mapping", resolved)

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load character mapping from a local CSV or JSON file."""
        return cls(mapping=load_mapping_file(path))

    @classmethod
    def from_session(
        cls,
        session: ModelSession,
        *,
        mapping_file: Path | str | None = None,
        hf_repo_id: str = DEFAULT_CHAR_IP_REPO,
        hf_filename: str = DEFAULT_CHAR_IP_FILE,
    ) -> Self:
        """Construct CharacterIPMapping by resolving from file, session artifacts, or Hugging Face [1]."""
        if not cls.is_supported(session):
            raise TransformRequirementError(
                f"Model '{session.model_id}' does not output character tags required for CharacterIPMapping."
            )

        # 1. Explicit manual file
        if mapping_file is not None:
            return cls.from_file(mapping_file)

        # 2. Check session artifacts for a model-bundled mapping file
        mapping_artifact = session.artifacts.get_optional("char_mapping") or session.artifacts.get_optional("mapping")
        if mapping_artifact is not None and mapping_artifact.is_file():
            logger.debug("Loading character IP mapping from session artifact: %s", mapping_artifact)
            return cls.from_file(mapping_artifact)

        # 3. Fallback to default community mapping from HF
        resolved = _resolve_default_mapping(hf_repo_id=hf_repo_id, hf_filename=hf_filename)
        return cls(mapping=resolved)

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
            tag_name = entry.tag
            # Match against both underscores and spaces
            norm_key = tag_name.replace(" ", "_")
            ips = self.mapping.get(norm_key) or self.mapping.get(tag_name)
            if ips:
                detected_mappings[tag_name] = list(ips)

        if not detected_mappings:
            return result

        extras = dict(result.extras)
        extras["character_copyright_mapping"] = detected_mappings
        return TagResult(categories=dict(result.categories), extras=extras)
