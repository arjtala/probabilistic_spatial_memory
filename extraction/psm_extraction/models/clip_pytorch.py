"""CLIP runner backed by HuggingFace transformers + PyTorch.

Auto-picks the best available device (cuda > mps > cpu) unless the caller
overrides via the `device` kwarg. Lifts the image/text encoding logic from
`scripts/e5_clip_demo.py` so the demo can become a thin shim over this
package.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .base import ModelRunner

DEFAULT_CHECKPOINT = "openai/clip-vit-base-patch32"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_device(requested: str) -> "object":
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CLIPPyTorchRunner(ModelRunner):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "auto",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "CLIPPyTorchRunner requires the [clip] extra. Install with: "
                "pip install -e extraction[clip]"
            ) from exc

        self._torch = __import__("torch")
        self.checkpoint = checkpoint
        self.model_id = "openai/clip"
        self._device = _resolve_device(device)
        self.backend = f"pytorch-{self._device.type}"
        self._processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
        self._model = CLIPModel.from_pretrained(checkpoint).eval().to(self._device)
        self.embedding_dim = int(self._model.config.projection_dim)
        self.normalized = True
        self.preprocess = "clip_default(resize=224,center_crop=224,normalize=clip)"
        self.patch_grid = None

    def embed_images(
        self,
        paths: Sequence[Path],
        batch_size: int = 16,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        if not paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        n_total = len(paths)
        with torch.inference_mode():
            for start in range(0, n_total, batch_size):
                batch = [
                    Image.open(p).convert("RGB") for p in paths[start : start + batch_size]
                ]
                try:
                    inputs = self._processor(images=batch, return_tensors="pt")
                finally:
                    for img in batch:
                        img.close()
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                feats = self._model.get_image_features(**inputs)
                if self.normalized:
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                chunks.append(feats.detach().cpu().to(torch.float32).numpy())
                if progress is not None:
                    progress(min(start + batch_size, n_total))
        return np.concatenate(chunks, axis=0)

    def embed_text(self, query: str) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            inputs = self._processor(text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._model.get_text_features(**inputs)
            if self.normalized:
                feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].detach().cpu().to(torch.float32).numpy().astype(np.float32)

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
