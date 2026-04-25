"""Model runner protocol + registry for the extraction pipeline."""

from .base import ModelRunner
from .registry import SUPPORTED_FAMILIES, make_runner, resolve_backend

__all__ = ["ModelRunner", "SUPPORTED_FAMILIES", "make_runner", "resolve_backend"]
