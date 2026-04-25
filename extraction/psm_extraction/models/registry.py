"""Backend resolution + runner factory.

Auto-picks one backend in priority order:
    CUDA-PyTorch  >  MLX-on-Apple-Silicon  >  MPS-PyTorch  >  CPU-PyTorch

Until the MLX runner is implemented (see clip_mlx.py), Apple Silicon falls
through to MPS-PyTorch. Forcing the choice is always possible via
`backend={'pytorch','mlx','cpu'}`.
"""

import platform
from typing import Any

from .base import ModelRunner


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    # The runner itself is currently a stub; the auto-pick should not surface
    # MLX until the implementation lands (see clip_mlx.py).
    try:
        from .clip_mlx import CLIPMLXRunner

        CLIPMLXRunner.__init__.__doc__  # touch
    except Exception:  # noqa: BLE001
        return False
    try:
        CLIPMLXRunner()
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _has_torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_backend(requested: str) -> str:
    """Translate 'auto' into one of {'pytorch', 'mlx', 'cpu'}.

    Non-auto inputs are passed through after validation. 'cpu' is a shorthand
    for forcing pytorch+cpu (useful for tests).
    """
    if requested == "auto":
        if _has_torch_cuda():
            return "pytorch"
        if _is_apple_silicon() and _has_mlx():
            return "mlx"
        return "pytorch"
    if requested in {"pytorch", "mlx", "cpu"}:
        return requested
    raise ValueError(
        f"unknown backend {requested!r}; expected one of: auto, pytorch, mlx, cpu"
    )


def make_runner(
    name: str,
    *,
    checkpoint: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    **kwargs: Any,
) -> ModelRunner:
    """Construct a runner for the named model family.

    name: short family identifier. For Phase 2, only 'clip' is supported.
    checkpoint: HF / hub identifier; defaults to the family's canonical small
        variant when omitted.
    backend: see resolve_backend().
    device: only consulted by PyTorch backends. 'auto' picks cuda > mps > cpu.
    """
    if name != "clip":
        raise ValueError(f"unsupported model family: {name!r}; supported: ['clip']")

    chosen = resolve_backend(backend)
    if chosen == "cpu":
        from .clip_pytorch import CLIPPyTorchRunner

        return CLIPPyTorchRunner(
            checkpoint=checkpoint or "openai/clip-vit-base-patch32",
            device="cpu",
            **kwargs,
        )
    if chosen == "mlx":
        from .clip_mlx import CLIPMLXRunner

        return CLIPMLXRunner(
            checkpoint=checkpoint or "openai/clip-vit-base-patch32",
            **kwargs,
        )
    from .clip_pytorch import CLIPPyTorchRunner

    return CLIPPyTorchRunner(
        checkpoint=checkpoint or "openai/clip-vit-base-patch32",
        device=device,
        **kwargs,
    )
