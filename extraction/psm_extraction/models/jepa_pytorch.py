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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import EmbedResult, ModelRunner
from .clip_pytorch import _resolve_device

if TYPE_CHECKING:
    import torch  # noqa: F401

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
        device_obj: "torch.device" = _resolve_device(device)
        self._device = device_obj
        self.backend = f"pytorch-{device_obj.type}"

        try:
            processor = AutoVideoProcessor.from_pretrained(checkpoint)
            if processor is None:
                raise RuntimeError(
                    f"AutoVideoProcessor returned None for {checkpoint!r}"
                )
            self._processor = processor
            self._model = (
                AutoModel.from_pretrained(checkpoint).eval().to(device_obj)
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

    def _safe_batch_size(self, requested: int) -> int:
        """Cap batch_size so the SDPA attention matrix doesn't OOM.

        V-JEPA 2 attention is O((batch * fpc * spatial_patches)^2) in memory;
        for ViT-L/16 at 256x256 (256 spatial patches) and fpc=64 a single batch
        already produces a 16384-token sequence, and even batch=2 explodes past
        a reasonable MPS budget. Cap based on fpc.
        """
        if self._fpc >= 64:
            return max(1, min(requested, 1))
        if self._fpc >= 32:
            return max(1, min(requested, 2))
        if self._fpc >= 16:
            return max(1, min(requested, 4))
        return max(1, min(requested, 8))

    def embed_images(
        self,
        paths: Sequence[Path],
        batch_size: int = 4,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Embed each frame by replicating it to fill the clip window.

        `batch_size` is clamped per `_safe_batch_size` so the orchestrator's
        default (16) doesn't trigger a multi-GB SDPA allocation. With the
        canonical `vjepa2-vitl-fpc64-256` checkpoint the effective batch is 1
        regardless of the requested value.
        """
        import sys

        from PIL import Image

        torch = self._torch
        if not paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), None

        safe_batch = self._safe_batch_size(batch_size)
        if safe_batch != batch_size:
            print(
                f"VJEPAPyTorchRunner: clamping batch_size {batch_size}->{safe_batch} "
                f"to avoid OOM (fpc={self._fpc}, attention scales quadratically)",
                file=sys.stderr,
            )
        batch_size = safe_batch
        n_total = len(paths)

        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, n_total, batch_size):
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
                if progress is not None:
                    progress(min(start + batch_size, n_total))

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
