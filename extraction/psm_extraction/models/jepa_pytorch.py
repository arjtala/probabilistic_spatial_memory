"""V-JEPA 2 PyTorch runner via HuggingFace transformers.

V-JEPA 2 lives on the HF Hub (e.g. `facebook/vjepa2-vitl-fpc64-256`) and is
loadable via `AutoModel.from_pretrained` once `transformers >= 4.53`. Note
the processor type is `AutoVideoProcessor` rather than `AutoImageProcessor`
— the model's input pipeline expects a sequence of frames per sample, not
a single image.

Per the upstream model card, the documented way to embed a still image is to
replicate it across the clip's frame count (e.g. 64 frames). That keeps this
runner compatible with the rest of the extraction pipeline (one ffmpeg pass
producing per-frame JPEGs) at the cost of running an N-frame forward pass
per sampled frame. On Apple Silicon this is roughly 1-2s per call for the
ViT-Large variant — pick a sparse `--sample-fps` (e.g. 0.5–1.0) for any
multi-minute video.

Prediction maps (the per-patch error against the target encoder, reshaped
to a 16×16 grid in the existing Aria pipeline) are not produced here; the
JEPA prediction objective requires sampling context+target patches and
running both the encoder and predictor heads, which is more involved than
the encoder-only path. Tracked as a Phase 4 follow-up.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import ModelRunner
from .clip_pytorch import _resolve_device

DEFAULT_CHECKPOINT = "facebook/vjepa2-vitl-fpc64-256"
FALLBACK_FPC = 16

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_frames_per_clip(config, checkpoint: str) -> int:
    """Determine how many frames the V-JEPA 2 processor expects per sample.

    Tries config attributes first (the V-JEPA 2 config exposes one of
    `frames_per_clip` / `num_frames` / `tubelet_size`), then falls back to
    parsing the checkpoint identifier (HF names embed `fpc16` / `fpc64`).
    """
    for attr in ("frames_per_clip", "num_frames"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    for part in checkpoint.lower().replace("/", "-").split("-"):
        if part.startswith("fpc") and part[3:].isdigit():
            return int(part[3:])
    return FALLBACK_FPC


class VJEPAPyTorchRunner(ModelRunner):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "auto",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VJEPAPyTorchRunner requires the [clip] extra plus a "
                "transformers version with V-JEPA 2 support (>= 4.53). "
                "Install with:\n"
                "  pip install -e extraction[clip]\n"
                "  pip install -U 'transformers>=4.53'"
            ) from exc

        self._torch = __import__("torch")
        self.checkpoint = checkpoint
        self.model_id = "facebookresearch/v-jepa-2"
        self._device = _resolve_device(device)
        self.backend = f"pytorch-{self._device.type}"

        try:
            self._processor = AutoVideoProcessor.from_pretrained(checkpoint)
            self._model = (
                AutoModel.from_pretrained(checkpoint).eval().to(self._device)
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"failed to load V-JEPA 2 checkpoint {checkpoint!r}: {exc}\n"
                f"Tip: V-JEPA 2 needs `transformers >= 4.53` (or the preview "
                f"branch). Run: pip install -U 'transformers>=4.53'"
            ) from exc

        cfg = self._model.config
        self.embedding_dim = int(getattr(cfg, "hidden_size", 1024))
        self.normalized = False
        self.preprocess = "vjepa2_autovideoprocessor"
        # Prediction maps deferred to Phase 4; encoder embeddings only here.
        self.patch_grid = None
        self._fpc = _resolve_frames_per_clip(cfg, checkpoint)

    @property
    def frames_per_clip(self) -> int:
        return self._fpc

    def embed_images(
        self, paths: Sequence[Path], batch_size: int = 4
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Embed each frame by replicating it to fill the clip window.

        Default `batch_size=4` is intentionally small — each "video" passed to
        the processor expands to `fpc` frames, so a batch of 4 with `fpc=64`
        already pushes 256 frames through the encoder per forward pass.
        """
        from PIL import Image

        torch = self._torch
        if not paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), None

        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                opened = [Image.open(p).convert("RGB") for p in batch_paths]
                try:
                    # V-JEPA 2 expects a sequence per "video"; replicate the
                    # static image to fill the model's clip window. The
                    # upstream model card explicitly endorses this pattern
                    # for image embedding.
                    videos = [[img] * self._fpc for img in opened]
                    inputs = self._processor(videos=videos, return_tensors="pt")
                finally:
                    for img in opened:
                        img.close()
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state  # (B, tokens, D)
                # Mean-pool every encoder token. V-JEPA 2 doesn't have a CLS
                # token, so averaging the full sequence gives the global
                # representation per clip.
                pooled = hidden.mean(dim=1)
                embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        emb = np.concatenate(embeddings, axis=0)
        return emb, None

    def embed_text(self, query: str) -> np.ndarray:
        raise NotImplementedError(
            "V-JEPA 2 is a vision-only encoder; use the CLIP runner for text queries"
        )

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
