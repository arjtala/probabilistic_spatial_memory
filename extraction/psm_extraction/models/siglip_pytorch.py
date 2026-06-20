"""SigLIP 2 runner backed by HuggingFace transformers + PyTorch.

Sibling to clip_pytorch.py for Google's SigLIP 2 family of dual-tower
vision-language encoders. Different training recipe and loss from
laion CLIP (sigmoid-pairwise rather than softmax-contrastive,
WebLI-100B rather than LAION-2B), so a useful non-CLIP encoder
ablation for the §5 corpus story.

API differences from CLIPPyTorchRunner that this file accommodates:

  - `AutoModel.from_pretrained(...)`, not `CLIPModel.from_pretrained(...)`.
    HF's auto-class loads SiglipModel, which has the same
    get_image_features / get_text_features surface but returns a
    `BaseModelOutputWithPooling` object — we pull `.pooler_output`
    (shape (B, hidden_size)) to match CLIP's bare-tensor convention.
  - No `projection_dim` config attribute: SigLIP's image and text
    pooled outputs are already in a shared space at `hidden_size`
    (1024 for siglip2-large-patch16-256). embedding_dim is read
    from config.vision_config.hidden_size.
  - Tokenizer max_length is 64 (vs CLIP's 77). Same truncation
    rationale applies for long narrations (Nymeria atomic_action,
    Ego-Exo4D atomic_descriptions, LookOut Gemini captions): drop
    trailing context rather than fail.
  - Image processor expects 256×256 input, not 224×224. The
    `preprocess` attr is updated to reflect that.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import ModelRunner

# Most useful default: same scale-tier as laion CLIP-ViT-L (660M vs
# 430M parameters; embedding dim 1024 vs 768) so the cross-encoder
# comparison is apples-to-apples in compute envelope. Other options:
#   - google/siglip2-base-patch16-256       (370M, 768-d)
#   - google/siglip2-giant-opt-patch16-256  (1.9B, 1536-d)
DEFAULT_CHECKPOINT = "google/siglip2-large-patch16-256"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_device(requested: str):
    """Untyped on purpose — torch is an optional runtime dependency."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pooled(out, model, modality: str):
    """Extract the pooled image/text embedding from SigLIP's output.

    SigLIP's `get_image_features` / `get_text_features` always return
    `BaseModelOutputWithPooling` (whereas CLIP returns a bare tensor
    in some transformers versions). We always pull `.pooler_output`
    which is shape `(B, hidden_size)` — the projection-equivalent for
    SigLIP, since image and text towers already share the embedding
    dimension.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out  # defensive: future API change might return a tensor
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        # Fallback: mean-pool the last hidden state. Should never
        # trigger on a healthy SigLIP install but documents intent.
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(
        f"SigLIP {modality} forward returned an unexpected type: {type(out).__name__}"
    )


class SiglipPyTorchRunner(ModelRunner):
    # See clip_pytorch.py for the pyrefly rationale on `Any` annotations.
    _device: Any
    _processor: Any
    _model: Any
    _torch: Any

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "auto",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SiglipPyTorchRunner requires the [clip] extra. Install with: "
                "pip install -e extraction[clip]"
            ) from exc

        self._torch = __import__("torch")
        self.checkpoint = checkpoint
        self.model_id = "google/siglip"
        self._device = _resolve_device(device)
        self.backend = f"pytorch-{self._device.type}"
        processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
        if processor is None:
            raise RuntimeError(f"AutoProcessor returned None for {checkpoint!r}")
        self._processor = processor
        self._model = AutoModel.from_pretrained(checkpoint).eval().to(self._device)

        # SigLIP keeps image + text towers in a shared embedding space
        # at vision_config.hidden_size (1024 for siglip2-large). No
        # separate projection_dim like CLIP.
        cfg = self._model.config
        hidden = getattr(getattr(cfg, "vision_config", None), "hidden_size", None)
        if hidden is None:
            raise RuntimeError(
                f"SigLIP {checkpoint!r} has no vision_config.hidden_size in its config"
            )
        self.embedding_dim = int(hidden)
        self.normalized = True

        # SigLIP 2 image processor: 256×256 patches, ImageNet
        # normalization (SigLIP-1 used 224×224; the 2-series jumped
        # patch resolution). Record the actual size from the processor
        # rather than guessing.
        try:
            size = self._processor.image_processor.size
            res = size.get("height") or size.get("shortest_edge") or 256
        except Exception:
            res = 256
        self.preprocess = f"siglip_default(resize={res},normalize=imagenet)"
        self.patch_grid = None

        # Text truncation cap: SigLIP-2 uses 64 tokens (vs CLIP's 77).
        # The processor's tokenizer.model_max_length is set to int.inf
        # for some HF builds, so pin from text_config.max_position_embeddings.
        text_cfg = getattr(cfg, "text_config", None)
        self._text_max_length = int(getattr(text_cfg, "max_position_embeddings", 64))

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
                out = self._model.get_image_features(**inputs)
                feats = _pooled(out, self._model, "image")
                if self.normalized:
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                chunks.append(feats.detach().cpu().to(torch.float32).numpy())
                if progress is not None:
                    progress(min(start + batch_size, n_total))
        return np.concatenate(chunks, axis=0)

    def embed_text(self, query: str) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            # SigLIP-2 text tower caps at 64 tokens. Long-form narrations
            # (Nymeria atomic_action, Gemini captions on LookOut) routinely
            # exceed this — drop trailing context rather than fail.
            inputs = self._processor(
                text=[query],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._text_max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            out = self._model.get_text_features(**inputs)
            feats = _pooled(out, self._model, "text")
            if self.normalized:
                feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].detach().cpu().to(torch.float32).numpy().astype(np.float32)

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
