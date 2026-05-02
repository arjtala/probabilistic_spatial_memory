"""MLX-native CLIP runner for Apple Silicon (stub for Phase 2 first cut).

Phase 2 focuses on getting the PyTorch path correct and the protocol clean.
The MLX backend is wired into the registry's auto-detection but the runner
itself currently raises NotImplementedError to make the unimplemented code
path explicit. A follow-up commit (tracked in TODO.md → "Extraction Pipeline"
→ Phase 2) will replace this body with a real MLX path once the upstream
mlx-clip API stabilizes.

Until then, on Apple Silicon the PyTorch MPS backend is the auto-pick.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .base import EmbedResult, ModelRunner


class CLIPMLXRunner(ModelRunner):
    def __init__(self, checkpoint: str = "openai/clip-vit-base-patch32") -> None:
        raise NotImplementedError(
            "MLX-native CLIP runner is not implemented yet; "
            "use backend='pytorch' (auto-picks MPS on Apple Silicon). "
            "See TODO.md → Extraction Pipeline → Phase 2 for the follow-up."
        )

    def embed_images(
        self,
        paths: Sequence[Path],
        batch_size: int = 16,
        *,
        progress: "Callable[[int], None] | None" = None,
    ) -> EmbedResult:  # pragma: no cover
        raise NotImplementedError

    def embed_text(self, query: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
