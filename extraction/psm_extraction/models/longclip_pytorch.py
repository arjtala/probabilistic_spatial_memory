"""Long-CLIP runner: drop-in CLIP replacement with a 248-token text encoder.

Long-CLIP (Zhang et al., ECCV 2024; github.com/beichenzbc/Long-CLIP)
extends CLIP's effective text input from ~77 tokens to 248 via
positional-embedding stretching + primary component matching. The
image encoder is unchanged. We use the L/14 checkpoint
(\"longclip-L.pt\") which matches the OpenAI CLIP-L architecture for
the image side, so existing image-embedding banks remain comparable.

This is the single highest-leverage encoder change for the Nymeria
benchmark: atomic_action narrations routinely tokenize to 80--120
tokens, exceeding standard CLIP's 77-token positional limit. Truncation
drops trailing context (\"...then opens the door with her right hand\")
that often contains the disambiguating action verb.

## Install

The official Long-CLIP repo is not on PyPI; install from source. From
the cluster (write access to your home dir):

    git clone https://github.com/beichenzbc/Long-CLIP.git ~/long-clip
    # Download the L/14 checkpoint per the repo's README (Google Drive
    # link in the README — ~1.7GB). Place at ~/long-clip/checkpoints/longclip-L.pt
    export LONGCLIP_ROOT=~/long-clip
    # The runner adds $LONGCLIP_ROOT to sys.path on import.

## Usage

    from psm_extraction.models import make_runner
    runner = make_runner(\"longclip\", device=\"cpu\")
    feats = runner.embed_text(\"long narration ...\")  # 768-d, 248-token cap

The text encoder dim matches OpenAI CLIP-L (768), so embeddings are
comparable to existing clip_l_features.h5 banks at the image side.
The text side is a non-trivial swap — re-extract or recompute text
embeddings at eval time when switching from CLIP-L to Long-CLIP.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .base import ModelRunner

DEFAULT_CHECKPOINT_FILENAME = "longclip-L.pt"
DEFAULT_CONTEXT_LENGTH = 248


def _resolve_device(device: str) -> Any:
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _ensure_longclip_on_path() -> None:
    """Add the Long-CLIP repo root to sys.path so its modules import."""
    root = os.environ.get("LONGCLIP_ROOT")
    if not root:
        raise RuntimeError(
            "LONGCLIP_ROOT is not set. Clone the Long-CLIP repo and "
            "export LONGCLIP_ROOT=<repo path>. See longclip_pytorch.py's "
            "module docstring for the full install steps."
        )
    p = Path(root).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"LONGCLIP_ROOT={p} does not exist")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class LongCLIPPyTorchRunner(ModelRunner):
    """Long-CLIP runner. Same ModelRunner interface as CLIPPyTorchRunner."""

    _model: Any
    _preprocess: Any
    _torch: Any

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        device: str = "auto",
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LongCLIPPyTorchRunner requires torch. Install with: "
                "pip install -e extraction[clip]"
            ) from exc
        self._torch = __import__("torch")

        _ensure_longclip_on_path()
        try:
            from longclip import load as longclip_load
            from longclip import tokenize as longclip_tokenize
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the `longclip` package. Ensure the "
                "Long-CLIP repo is cloned and LONGCLIP_ROOT points to "
                "its root directory."
            ) from exc

        self._tokenize = longclip_tokenize
        self.context_length = int(context_length)

        # Default checkpoint: look in $LONGCLIP_ROOT/checkpoints/longclip-L.pt
        if checkpoint is None:
            root = Path(os.environ["LONGCLIP_ROOT"]).expanduser().resolve()
            checkpoint = str(root / "checkpoints" / DEFAULT_CHECKPOINT_FILENAME)
        self.checkpoint = checkpoint
        self.model_id = "longclip-L-vit-l-14"
        self._device = _resolve_device(device)
        self.backend = f"pytorch-{self._device.type}"

        # longclip.load returns (model, preprocess); model has .encode_image,
        # .encode_text methods (OpenAI-CLIP style API).
        model, preprocess = longclip_load(checkpoint, device=str(self._device))
        model.eval()
        self._model = model
        self._preprocess = preprocess

        # Long-CLIP-L uses the OpenAI CLIP-L vision tower → 768-d projection.
        # Embed a dummy to discover the dim rather than hard-coding.
        with self._torch.inference_mode():
            tok = self._tokenize(["probe"], context_length=self.context_length)
            tok = tok.to(self._device)
            probe = model.encode_text(tok)
        self.embedding_dim = int(probe.shape[-1])
        self.normalized = True
        self.preprocess = "longclip_default(resize=224,center_crop=224,normalize=clip)"
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
                imgs = []
                for p in paths[start : start + batch_size]:
                    with Image.open(p) as im:
                        imgs.append(self._preprocess(im.convert("RGB")))
                batch = torch.stack(imgs).to(self._device)
                feats = self._model.encode_image(batch)
                if self.normalized:
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                chunks.append(feats.detach().cpu().to(torch.float32).numpy())
                if progress is not None:
                    progress(min(start + batch_size, n_total))
        return np.concatenate(chunks, axis=0)

    def embed_text(self, query: str) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            tok = self._tokenize([query], context_length=self.context_length)
            tok = tok.to(self._device)
            feats = self._model.encode_text(tok)
            if self.normalized:
                feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].detach().cpu().to(torch.float32).numpy().astype(np.float32)

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
