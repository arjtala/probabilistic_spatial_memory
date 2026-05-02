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


def _resolve_device(requested: str):
    """Return a `torch.device`. Untyped on purpose — `torch` is an optional
    runtime dependency; importing it eagerly for the annotation would defeat
    the [clip] extra. Callers treat the return value duck-typed (`.type`
    attribute, passable to `.to()`)."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _coerce_clip_features(out, model, modality: str):
    # transformers 5.x's CLIPModel.get_{image,text}_features returns a
    # BaseModelOutputWithPooling whose pooler_output is already the
    # projected embedding (shape == projection_dim). Older versions
    # returned a tensor directly. Either way, only project further when
    # the shape clearly comes from the pre-projection encoder hidden
    # state (last dim == hidden_size != projection_dim).
    import torch

    config = getattr(model, "config", None)
    proj_dim = int(getattr(config, "projection_dim", -1)) if config is not None else -1
    if isinstance(out, torch.Tensor):
        tensor = out
    else:
        tensor = (
            getattr(out, "image_embeds", None)
            or getattr(out, "text_embeds", None)
            or getattr(out, "pooler_output", None)
        )
        if tensor is None:
            last = getattr(out, "last_hidden_state", None)
            if last is not None and last.dim() == 3:
                tensor = last[:, 0, :]
        if tensor is None:
            raise RuntimeError(
                f"Unexpected CLIP {modality} output shape: {type(out).__name__}"
            )
    if proj_dim > 0 and tensor.shape[-1] == proj_dim:
        return tensor
    projection = getattr(
        model,
        "visual_projection" if modality == "image" else "text_projection",
        None,
    )
    if projection is None:
        return tensor
    return projection(tensor)


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
                feats = _coerce_clip_features(feats, self._model, "image")
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
            feats = _coerce_clip_features(feats, self._model, "text")
            if self.normalized:
                feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].detach().cpu().to(torch.float32).numpy().astype(np.float32)

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
