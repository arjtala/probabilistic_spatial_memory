"""V-JEPA 2 PyTorch runner — Phase 3 first cut, encoder embeddings only.

The full V-JEPA 2 pipeline produces both encoder features and per-patch
prediction-error maps (16x16 in the existing Aria pipeline). The first cut
ships the encoder side only — an `AutoModel.from_pretrained` over the V-JEPA
2 checkpoint, with mean-pooled encoder tokens as the 1024-d embedding.

Prediction maps require running the predictor head against context+target
patches and computing per-patch error against the target encoder; deferred
to Phase 4 once the upstream model card stabilizes the inference recipe.

If the loader fails (the checkpoint isn't yet supported by AutoModel for
some HF transformers versions), the constructor surfaces a clear runtime
error suggesting that the user check their `transformers` version and
checkpoint identifier rather than silently mis-running.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import ModelRunner
from .clip_pytorch import _resolve_device

DEFAULT_CHECKPOINT = "facebook/vjepa2-vit-large"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class VJEPAPyTorchRunner(ModelRunner):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "auto",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "VJEPAPyTorchRunner requires the [clip] extra. "
                "Install with: pip install -e extraction[clip]"
            ) from exc

        self._torch = __import__("torch")
        self.checkpoint = checkpoint
        self.model_id = "facebookresearch/v-jepa-2"
        self._device = _resolve_device(device)
        self.backend = f"pytorch-{self._device.type}"

        try:
            self._processor = AutoImageProcessor.from_pretrained(checkpoint)
            self._model = (
                AutoModel.from_pretrained(checkpoint).eval().to(self._device)
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"failed to load V-JEPA 2 checkpoint {checkpoint!r}: {exc}. "
                f"V-JEPA 2 may need a custom loader; consider passing a "
                f"locally-cached HuggingFace checkpoint or pinning a "
                f"`transformers` version that exposes V-JEPA 2 via AutoModel. "
                f"See TODO.md → Extraction Pipeline → Phase 3."
            ) from exc

        cfg = self._model.config
        self.embedding_dim = int(getattr(cfg, "hidden_size", 1024))
        self.normalized = False
        self.preprocess = "vjepa2_autoimageprocessor"
        # Prediction maps deferred to Phase 4.
        self.patch_grid = None

    def embed_images(
        self, paths: Sequence[Path], batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray | None]:
        from PIL import Image

        torch = self._torch
        if not paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), None

        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch = [
                    Image.open(p).convert("RGB")
                    for p in paths[start : start + batch_size]
                ]
                try:
                    inputs = self._processor(images=batch, return_tensors="pt")
                finally:
                    for img in batch:
                        img.close()
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state  # (B, tokens, D)
                # Mean-pool every patch token; V-JEPA 2 doesn't have a CLS
                # token so we average the full sequence.
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
