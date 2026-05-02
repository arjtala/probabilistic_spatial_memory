"""HDF5 writer that emits schema-v2-compliant features.h5 files."""

import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from . import schema


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _validate_1d(name: str, array: np.ndarray, expected_len: int | None = None) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if expected_len is not None and array.shape[0] != expected_len:
        raise ValueError(
            f"{name} length {array.shape[0]} does not match expected {expected_len}"
        )


def _validate_2d(name: str, array: np.ndarray, n: int, dim: int) -> None:
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.shape != (n, dim):
        raise ValueError(
            f"{name} shape {array.shape} does not match expected ({n}, {dim})"
        )


class FeaturesWriter:
    """Context-managed writer for a single features.h5 file.

    Opens the file on construction (mode="w" by default), writes root-level
    schema-v2 metadata, and exposes one method per group kind. Use as a
    context manager (`with FeaturesWriter(...) as w:`) so the file closes
    deterministically even if a write step fails.

    Sensor groups (`gps`, `imu`) are the canonical raw sources of truth.
    Model groups (`dino`, `jepa`, `clip`, …) carry per-frame embeddings plus
    sensors interpolated onto frame timestamps for downstream convenience.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        mode: str = "w",
        producer_version: str = "0.1.0",
        source_video: str | None = None,
        session_id: str | None = None,
        created_at_utc: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._h: h5py.File | None = h5py.File(self._path, mode)
        self._h.attrs["schema_version"] = schema.SCHEMA_VERSION
        self._h.attrs["producer"] = schema.PRODUCER_NAME
        self._h.attrs["producer_version"] = producer_version
        self._h.attrs["created_at_utc"] = created_at_utc or _now_utc_iso()
        self._h.attrs["timestamp_unit"] = schema.DEFAULT_TIMESTAMP_UNIT
        self._h.attrs["coord_system"] = schema.DEFAULT_COORD_SYSTEM
        if source_video is not None:
            self._h.attrs["source_video"] = source_video
        if session_id is not None:
            self._h.attrs["session_id"] = session_id

    @property
    def path(self) -> Path:
        return self._path

    def write_gps_group(
        self,
        *,
        timestamps: np.ndarray,
        lat: np.ndarray,
        lng: np.ndarray,
        rate_hz_nominal: float | None = None,
    ) -> None:
        if "gps" in self._handle:
            raise RuntimeError("gps group already written")
        n = timestamps.shape[0]
        _validate_1d("gps/timestamps", timestamps)
        _validate_1d("gps/lat", lat, n)
        _validate_1d("gps/lng", lng, n)
        group = self._handle.create_group("gps")
        group.create_dataset("timestamps", data=timestamps.astype(np.float64))
        group.create_dataset("lat", data=lat.astype(np.float64))
        group.create_dataset("lng", data=lng.astype(np.float64))
        if rate_hz_nominal is not None:
            group.attrs["rate_hz_nominal"] = float(rate_hz_nominal)

    def write_imu_group(
        self,
        *,
        timestamps: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        rate_hz_nominal: float | None = None,
    ) -> None:
        if "imu" in self._handle:
            raise RuntimeError("imu group already written")
        n = timestamps.shape[0]
        _validate_1d("imu/timestamps", timestamps)
        _validate_2d("imu/accel", accel, n, 3)
        _validate_2d("imu/gyro", gyro, n, 3)
        group = self._handle.create_group("imu")
        group.create_dataset("timestamps", data=timestamps.astype(np.float64))
        group.create_dataset("accel", data=accel.astype(np.float32))
        group.create_dataset("gyro", data=gyro.astype(np.float32))
        if rate_hz_nominal is not None:
            group.attrs["rate_hz_nominal"] = float(rate_hz_nominal)

    def write_model_group(
        self,
        name: str,
        *,
        spec: schema.ModelGroupSpec,
        timestamps: np.ndarray,
        lat: np.ndarray,
        lng: np.ndarray,
        embeddings: np.ndarray,
        attention_maps: np.ndarray | None = None,
        prediction_maps: np.ndarray | None = None,
        accel: np.ndarray | None = None,
        gyro: np.ndarray | None = None,
    ) -> None:
        if name in {"gps", "imu"}:
            raise ValueError(f"'{name}' is reserved for sensor groups")
        if name in self._handle:
            raise RuntimeError(f"model group '{name}' already written")
        n = timestamps.shape[0]
        _validate_1d(f"{name}/timestamps", timestamps)
        _validate_1d(f"{name}/lat", lat, n)
        _validate_1d(f"{name}/lng", lng, n)
        _validate_2d(f"{name}/embeddings", embeddings, n, spec.embedding_dim)
        if attention_maps is not None:
            if attention_maps.ndim != 3 or attention_maps.shape[0] != n:
                raise ValueError(
                    f"{name}/attention_maps must have shape (N, h, w); got "
                    f"{attention_maps.shape}"
                )
        if prediction_maps is not None:
            if prediction_maps.ndim != 3 or prediction_maps.shape[0] != n:
                raise ValueError(
                    f"{name}/prediction_maps must have shape (N, h, w); got "
                    f"{prediction_maps.shape}"
                )
        if accel is not None:
            _validate_2d(f"{name}/accel", accel, n, 3)
        if gyro is not None:
            _validate_2d(f"{name}/gyro", gyro, n, 3)

        group = self._handle.create_group(name)
        group.create_dataset("timestamps", data=timestamps.astype(np.float64))
        group.create_dataset("lat", data=lat.astype(np.float64))
        group.create_dataset("lng", data=lng.astype(np.float64))
        group.create_dataset("embeddings", data=embeddings.astype(np.float32))
        if attention_maps is not None:
            group.create_dataset(
                "attention_maps", data=attention_maps.astype(np.float32)
            )
        if prediction_maps is not None:
            group.create_dataset(
                "prediction_maps", data=prediction_maps.astype(np.float32)
            )
        if accel is not None:
            group.create_dataset("accel", data=accel.astype(np.float32))
        if gyro is not None:
            group.create_dataset("gyro", data=gyro.astype(np.float32))

        group.attrs["model"] = spec.model
        group.attrs["checkpoint"] = spec.checkpoint
        group.attrs["embedding_dim"] = int(spec.embedding_dim)
        group.attrs["sample_fps"] = float(spec.sample_fps)
        group.attrs["sampling"] = spec.sampling
        group.attrs["preprocess"] = spec.preprocess
        group.attrs["normalized"] = bool(spec.normalized)
        if spec.patch_grid is not None:
            group.attrs["patch_grid"] = list(spec.patch_grid)
        if spec.interpolation is not None:
            group.attrs["interpolation"] = spec.interpolation

    def close(self) -> None:
        # Local-variable narrowing pattern: pyrefly does not narrow
        # `self._h` through the `is not None` guard alone (mutable attribute).
        h = self._h
        if h is not None:
            h.close()
            self._h = None

    def __enter__(self) -> "FeaturesWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def _handle(self) -> h5py.File:
        if self._h is None:
            raise RuntimeError("FeaturesWriter is closed")
        return self._h
