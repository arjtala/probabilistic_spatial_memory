"""Versioned schema constants and lightweight types for features.h5.

Schema v2 makes file-level metadata explicit (schema_version, producer,
created_at_utc, coord_system, timestamp_unit) and tightens per-model-group
attributes (model, checkpoint, embedding_dim, sample_fps, normalized).

Schema v1 is the pre-existing format consumed by the C ingest. The C ingest
only reads dataset arrays and ignores attrs, so it stays compatible with both
versions; v2 is purely additive metadata.
"""

from __future__ import annotations

import dataclasses
from typing import Final

SCHEMA_VERSION: Final[int] = 2
PRODUCER_NAME: Final[str] = "psm-extraction"
DEFAULT_TIMESTAMP_UNIT: Final[str] = "unix_seconds_f64"
DEFAULT_COORD_SYSTEM: Final[str] = "WGS84_degrees"

# Datasets the C ingest is known to read in a model group. Optional datasets
# (attention/prediction maps, per-frame imu) are written when supplied but
# never required by the engine.
MODEL_REQUIRED_DATASETS: Final[tuple[str, ...]] = (
    "timestamps",
    "lat",
    "lng",
    "embeddings",
)
MODEL_OPTIONAL_DATASETS: Final[tuple[str, ...]] = (
    "attention_maps",
    "prediction_maps",
    "accel",
    "gyro",
)

# Attrs every v2 model group is expected to carry. A v1→v2 migration may
# back-fill these with best-effort defaults for known group names; a fresh
# writer always sets them explicitly.
MODEL_REQUIRED_ATTRS: Final[tuple[str, ...]] = (
    "model",
    "checkpoint",
    "embedding_dim",
    "sample_fps",
    "sampling",
    "preprocess",
    "normalized",
)
MODEL_OPTIONAL_ATTRS: Final[tuple[str, ...]] = (
    "patch_grid",
    "interpolation",
)

ROOT_REQUIRED_ATTRS: Final[tuple[str, ...]] = (
    "schema_version",
    "producer",
    "producer_version",
    "created_at_utc",
    "timestamp_unit",
    "coord_system",
)
ROOT_OPTIONAL_ATTRS: Final[tuple[str, ...]] = (
    "source_video",
    "session_id",
)


@dataclasses.dataclass(frozen=True)
class ModelGroupSpec:
    """Required + optional metadata describing one model group.

    Used as the typed surface of `FeaturesWriter.write_model_group(...)`.
    """

    model: str
    checkpoint: str
    embedding_dim: int
    sample_fps: float
    sampling: str
    preprocess: str
    normalized: bool
    # Optional: only emit attrs when the producer actually has a meaningful
    # value. Patch grids only apply to groups that ship attention/prediction
    # maps; interpolation only applies to groups carrying interpolated
    # per-frame lat/lng/imu derived from the canonical sensor streams.
    patch_grid: tuple[int, int] | None = None
    interpolation: str | None = None
