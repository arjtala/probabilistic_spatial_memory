"""Model runner protocol + registry for the extraction pipeline."""

from .base import ModelRunner
from .registry import make_runner, resolve_backend

__all__ = ["ModelRunner", "make_runner", "resolve_backend"]
