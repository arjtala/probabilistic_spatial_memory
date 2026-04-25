"""Abstract base class every model runner implements.

The runner is responsible for its own preprocessing (resize, normalize, etc.)
and for producing L2-normalized embeddings unless `normalized` is False with a
documented reason. The orchestrator (`extract.py`) treats runners as opaque:
load → embed batches of frames → optionally embed a text query → close.
"""

import abc
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


class ModelRunner(abc.ABC):
    """Minimum surface every backend implements."""

    # Subclasses set these in __init__ before super().__init__() returns.
    model_id: str
    """Stable family name recorded in HDF5 attrs (e.g. 'openai/clip')."""

    checkpoint: str
    """Specific weights identifier (e.g. 'openai/clip-vit-base-patch32')."""

    embedding_dim: int
    """Dimensionality of the embedding vectors this runner produces."""

    normalized: bool
    """Whether the embeddings are L2-normalized to unit length."""

    preprocess: str
    """Human-readable description of the preprocessing pipeline."""

    patch_grid: tuple[int, int] | None
    """`(h, w)` patch grid for attention/prediction-map producers; None otherwise."""

    backend: str
    """Backend tag recorded for debugging (e.g. 'pytorch-mps', 'mlx')."""

    @abc.abstractmethod
    def embed_images(
        self,
        paths: Sequence[Path],
        batch_size: int = 16,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Return a `(N, embedding_dim)` float32 array for the given image paths.

        `progress`, when supplied, is called with the running count of frames
        completed so far. Runners should invoke it after each batch.
        """

    @abc.abstractmethod
    def embed_text(self, query: str) -> np.ndarray:
        """Return a `(embedding_dim,)` float32 array for the given text query."""

    def close(self) -> None:
        """Release model weights / GPU memory. Default: no-op."""

    def __enter__(self) -> "ModelRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
