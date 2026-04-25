"""Backend-resolution tests — no torch / transformers / mlx required."""

import sys
from unittest import mock

import pytest

from psm_extraction.models import registry


def test_resolve_backend_passes_through_explicit() -> None:
    assert registry.resolve_backend("pytorch") == "pytorch"
    assert registry.resolve_backend("mlx") == "mlx"
    assert registry.resolve_backend("cpu") == "cpu"


def test_resolve_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        registry.resolve_backend("tensorflow")


def test_resolve_backend_auto_picks_cuda_when_available() -> None:
    with mock.patch.object(registry, "_has_torch_cuda", return_value=True):
        with mock.patch.object(registry, "_is_apple_silicon", return_value=False):
            assert registry.resolve_backend("auto") == "pytorch"


def test_resolve_backend_auto_falls_back_to_pytorch_when_mlx_unavailable() -> None:
    with mock.patch.object(registry, "_has_torch_cuda", return_value=False):
        with mock.patch.object(registry, "_is_apple_silicon", return_value=True):
            with mock.patch.object(registry, "_has_mlx", return_value=False):
                assert registry.resolve_backend("auto") == "pytorch"


def test_resolve_backend_auto_picks_mlx_only_when_runner_works() -> None:
    with mock.patch.object(registry, "_has_torch_cuda", return_value=False):
        with mock.patch.object(registry, "_is_apple_silicon", return_value=True):
            with mock.patch.object(registry, "_has_mlx", return_value=True):
                assert registry.resolve_backend("auto") == "mlx"


def test_make_runner_rejects_unsupported_family() -> None:
    with pytest.raises(ValueError, match="unsupported model family"):
        registry.make_runner("vmamba")


def test_make_runner_with_mlx_raises_until_implemented() -> None:
    # Until clip_mlx.py grows a real implementation, requesting it explicitly
    # surfaces NotImplementedError so callers can fall back.
    with pytest.raises(NotImplementedError):
        registry.make_runner("clip", backend="mlx")
