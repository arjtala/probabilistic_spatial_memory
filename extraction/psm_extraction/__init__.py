"""Extraction pipeline for PSM features.h5 files (schema v2)."""

from .schema import (
    SCHEMA_VERSION,
    PRODUCER_NAME,
    DEFAULT_TIMESTAMP_UNIT,
    DEFAULT_COORD_SYSTEM,
    ModelGroupSpec,
)
from .writer import FeaturesWriter
from .migrate import migrate_v1_to_v2

__all__ = [
    "SCHEMA_VERSION",
    "PRODUCER_NAME",
    "DEFAULT_TIMESTAMP_UNIT",
    "DEFAULT_COORD_SYSTEM",
    "ModelGroupSpec",
    "FeaturesWriter",
    "migrate_v1_to_v2",
]

__version__ = "0.1.0"
