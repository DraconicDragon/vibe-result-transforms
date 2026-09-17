# vibe-result-transforms

Post-processing transforms for `vibe` vision model outputs.

`vibe` returns complete, raw predictions directly from models. This library handles the downstream filtering, tag cleaning, and enrichment.

## Installation

```bash
pip install vibe-result-transforms
```

## Quickstart

```python
import vibe
from vibe_result_transforms import CleanTags, ScoreThresholds, TransformPipeline

# 1. Run model inference in vibe
session = vibe.load("wd-eva02-large-v3")
raw_result = session.infer("image.png").first()

# 2. Build a transform pipeline
# fallback_threshold is required as a safety net if a model has no recommendations
pipeline = TransformPipeline([
    ScoreThresholds.from_session(session, fallback_threshold=0.35),
    CleanTags(),
])

# 3. Apply to raw result
processed = pipeline(raw_result)
print(processed.tag_names())
```

## Available Transforms

### `ScoreThresholds`

Filters tags using a base threshold and optional per-category overrides.

```python
from vibe_result_transforms import ScoreThresholds

# Uses model recommendations, falling back to 0.35 if any category lacks a recommendation
filter_tf = ScoreThresholds.from_session(session, fallback_threshold=0.35)

# Or set custom thresholds manually without a session
filter_tf = ScoreThresholds(
    threshold=0.35,
    category_thresholds={"character": 0.75, "rating": 0.5}
)

filtered_result = filter_tf(raw_result)
```

### `TagLevelThresholds`

Filters tags using per-tag calibrated decision thresholds (e.g. JTP Hydra or AnimeTimm).

```python
from vibe_result_transforms import TagLevelThresholds

tag_thresh = TagLevelThresholds.from_session(session, fallback=0.35)
filtered_result = tag_thresh(raw_result)
```

### `CleanTags`

Replaces underscores with spaces while preserving kaomojis (`^_^`, `o_o`, `>_<`, etc.).

```python
from vibe_result_transforms import CleanTags

cleaner = CleanTags()
cleaned_result = cleaner(raw_result)
```

### `CharacterIPMapping`

Maps character tags to their corresponding series / copyright tags (stored in `result.extras["character_copyright_mapping"]`).

```python
from vibe_result_transforms import CharacterIPMapping

mapper = CharacterIPMapping.from_session(session)
mapped_result = mapper(raw_result)
```

## UI Discovery

For frontend applications (Qt, React, Gradio), all transforms provide metadata and OpenAPI/JSON schemas [1, 2]:

```python
from vibe_result_transforms import list_transforms

for tf_cls in list_transforms():
    print(tf_cls.id, tf_cls.display_name)
    schema = tf_cls.get_schema()  # Standard JSON Schema for UI sliders/inputs
```
