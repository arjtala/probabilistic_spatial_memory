"""DINO (v2 / v3) PyTorch runner via HuggingFace transformers.

DINOv2 ships with a first-class HF transformers `Dinov2Model` class. DINOv3
weights load through the same `AutoModel` path when a corresponding
`Dinov3Model` is registered; for unreleased / custom DINOv3 checkpoints, the
caller can pass a local path.

The runner emits:
- `embeddings` — 1024-d for ViT-large, 768-d for ViT-base (probed from
  `model.config.hidden_size`). Mean-pooled patch tokens (excluding the CLS
  token) so the vector represents image content rather than a learned
  registration token.
- `attention_maps` — last-layer CLS-to-patch attention averaged across heads
  and reshaped to the patch grid. Shape `(N, h, w)` with `h * w` matching
  `patch_grid`.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .base import ModelRunner
from .clip_pytorch import _resolve_device

DEFAULT_CHECKPOINT = "facebook/dinov2-base"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _model_id_from_checkpoint(checkpoint: str) -> str:
    ck = checkpoint.lower()
    if "dinov3" in ck or "dinov-3" in ck or "dino-v3" in ck:
        return "facebookresearch/dinov3"
    return "facebookresearch/dinov2"


class DINOPyTorchRunner(ModelRunner):
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
                "DINOPyTorchRunner requires the [clip] extra (torch + transformers). "
                "Install with: pip install -e extraction[clip]"
            ) from exc

        self._torch = __import__("torch")
        self.checkpoint = checkpoint
        self.model_id = _model_id_from_checkpoint(checkpoint)
        self._device = _resolve_device(device)
        self.backend = f"pytorch-{self._device.type}"

        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = (
            AutoModel.from_pretrained(checkpoint, output_attentions=True)
            .eval()
            .to(self._device)
        )

        cfg = self._model.config
        self.embedding_dim = int(cfg.hidden_size)
        self.normalized = False
        # The HF AutoImageProcessor's defaults match the reference DINO setup
        # (resize to a model-specific short side + centre-crop + ImageNet
        # normalization). Recording the descriptor lets a reader audit which
        # preprocessing path produced the embeddings.
        self.preprocess = "dinov2_autoimageprocessor"
        # DINOv3 (and DINOv2-with-registers) prepend `num_register_tokens`
        # register tokens after the CLS token, before the patch tokens. The
        # config exposes this; defaults to 0 for plain DINOv2.
        self._num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        # Probed lazily on the first batch from the attention shape.
        self.patch_grid: tuple[int, int] | None = None

    def _probe_patch_grid(self, attentions_shape: tuple[int, ...]) -> tuple[int, int]:
        # attentions[-1] has shape (B, heads, tokens, tokens). For a typical
        # CLS-prefixed ViT, tokens = 1 + num_register + h*w where num_register
        # is 0 for DINOv2-base and 4 for DINOv3 / DINOv2-with-registers. We
        # subtract the known register count first; fall back to brute-forcing
        # plausible counts if the config didn't expose one.
        tokens = int(attentions_shape[-1])
        for r in (self._num_register_tokens, 0, 1, 2, 4, 8):
            n_patches = tokens - 1 - r
            if n_patches <= 0:
                continue
            side = int(round(n_patches**0.5))
            if side > 0 and side * side == n_patches:
                if r != self._num_register_tokens:
                    self._num_register_tokens = r
                return (side, side)
        raise RuntimeError(
            f"unexpected DINO attention token count {tokens}; "
            f"can't infer a square patch grid (tried register counts "
            f"0,1,2,4,8 around config.num_register_tokens="
            f"{self._num_register_tokens})"
        )

    def embed_images(
        self,
        paths: Sequence[Path],
        batch_size: int = 16,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return `(embeddings, attention_maps)`.

        `attention_maps` is the per-frame CLS-to-patch attention (averaged
        across heads) reshaped to the probed patch grid; None when the model
        config does not expose attentions for this checkpoint.
        """
        from PIL import Image

        torch = self._torch
        if not paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), None

        embeddings: list[np.ndarray] = []
        attentions: list[np.ndarray] = []
        n_total = len(paths)
        with torch.inference_mode():
            for start in range(0, n_total, batch_size):
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
                outputs = self._model(**inputs, output_attentions=True)
                attn = getattr(outputs, "attentions", None)
                if attn is not None and len(attn) > 0 and self.patch_grid is None:
                    # Probe up-front so we know how many register tokens to
                    # skip in the embedding pool below.
                    self.patch_grid = self._probe_patch_grid(tuple(attn[-1].shape))

                # Mean-pool patch tokens, excluding CLS + register tokens.
                hidden = outputs.last_hidden_state  # (B, tokens, D)
                patch_start = 1 + self._num_register_tokens
                patch_features = hidden[:, patch_start:, :].mean(dim=1)  # (B, D)
                embeddings.append(
                    patch_features.detach().cpu().to(torch.float32).numpy()
                )

                if attn is not None and len(attn) > 0:
                    last = attn[-1]  # (B, heads, tokens, tokens)
                    # patch_grid was probed above on the same `attn`, so it
                    # cannot be None here.
                    assert self.patch_grid is not None
                    h, w = self.patch_grid
                    # CLS-to-patch attention, skipping register tokens too,
                    # averaged across heads.
                    cls_to_patch = last[:, :, 0, patch_start:].mean(dim=1)
                    cls_to_patch = cls_to_patch.reshape(-1, h, w)
                    attentions.append(
                        cls_to_patch.detach().cpu().to(torch.float32).numpy()
                    )

                if progress is not None:
                    progress(min(start + batch_size, n_total))

        emb = np.concatenate(embeddings, axis=0)
        if attentions:
            attn_array: np.ndarray | None = np.concatenate(attentions, axis=0)
        else:
            attn_array = None
        return emb, attn_array

    def embed_text(self, query: str) -> np.ndarray:
        raise NotImplementedError(
            "DINO is a vision-only encoder; use the CLIP runner for text queries"
        )

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
