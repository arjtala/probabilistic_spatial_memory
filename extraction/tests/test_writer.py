"""Round-trip tests for FeaturesWriter."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from psm_extraction import schema
from psm_extraction.writer import FeaturesWriter


def test_writer_emits_root_attrs(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    with FeaturesWriter(out, source_video="data.mp4", session_id="abc") as _w:
        pass

    with h5py.File(out, "r") as h:
        assert int(h.attrs["schema_version"]) == schema.SCHEMA_VERSION
        assert str(h.attrs["producer"]) == schema.PRODUCER_NAME
        assert str(h.attrs["timestamp_unit"]) == schema.DEFAULT_TIMESTAMP_UNIT
        assert str(h.attrs["coord_system"]) == schema.DEFAULT_COORD_SYSTEM
        assert str(h.attrs["source_video"]) == "data.mp4"
        assert str(h.attrs["session_id"]) == "abc"
        assert "created_at_utc" in h.attrs


def test_writer_round_trips_gps(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    ts = np.linspace(100.0, 200.0, num=11)
    lat = np.full(11, 51.5, dtype=np.float64)
    lng = np.full(11, -0.1, dtype=np.float64)
    with FeaturesWriter(out) as w:
        w.write_gps_group(timestamps=ts, lat=lat, lng=lng, rate_hz_nominal=1.0)

    with h5py.File(out, "r") as h:
        assert h["gps/timestamps"].shape == (11,)
        assert h["gps/timestamps"].dtype == np.float64
        np.testing.assert_array_equal(h["gps/lat"][...], lat)
        np.testing.assert_array_equal(h["gps/lng"][...], lng)
        assert float(h["gps"].attrs["rate_hz_nominal"]) == 1.0


def test_writer_round_trips_imu(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    n = 1024
    ts = np.linspace(0.0, 1.0, num=n)
    accel = np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32)
    gyro = np.random.default_rng(1).standard_normal((n, 3)).astype(np.float32)
    with FeaturesWriter(out) as w:
        w.write_imu_group(
            timestamps=ts, accel=accel, gyro=gyro, rate_hz_nominal=1000.0
        )

    with h5py.File(out, "r") as h:
        assert h["imu/accel"].shape == (n, 3)
        assert h["imu/accel"].dtype == np.float32
        np.testing.assert_allclose(h["imu/gyro"][...], gyro)
        assert float(h["imu"].attrs["rate_hz_nominal"]) == 1000.0


def test_writer_round_trips_model_group_with_attention_maps(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    n = 8
    dim = 1024
    ts = np.arange(n, dtype=np.float64)
    lat = np.full(n, 37.0)
    lng = np.full(n, -122.0)
    embeddings = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    attention = np.random.default_rng(1).standard_normal((n, 14, 14)).astype(np.float32)

    spec = schema.ModelGroupSpec(
        model="facebookresearch/dinov3",
        checkpoint="dinov3-base-patch16-224",
        embedding_dim=dim,
        sample_fps=30.0,
        sampling="every_video_frame",
        preprocess="resize=224,center_crop=224,normalize=imagenet",
        normalized=True,
        patch_grid=(14, 14),
        interpolation="linear,nearest_at_edges,from=imu+gps",
    )
    with FeaturesWriter(out) as w:
        w.write_model_group(
            "dino",
            spec=spec,
            timestamps=ts,
            lat=lat,
            lng=lng,
            embeddings=embeddings,
            attention_maps=attention,
        )

    with h5py.File(out, "r") as h:
        group = h["dino"]
        assert group["embeddings"].shape == (n, dim)
        assert group["embeddings"].dtype == np.float32
        assert group["attention_maps"].shape == (n, 14, 14)
        assert int(group.attrs["embedding_dim"]) == dim
        assert float(group.attrs["sample_fps"]) == 30.0
        assert bool(group.attrs["normalized"]) is True
        assert list(group.attrs["patch_grid"]) == [14, 14]
        assert "interpolation" in group.attrs


def test_writer_rejects_shape_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    spec = schema.ModelGroupSpec(
        model="m", checkpoint="c", embedding_dim=4, sample_fps=1.0,
        sampling="s", preprocess="p", normalized=True,
    )
    with FeaturesWriter(out) as w:
        with pytest.raises(ValueError, match="embeddings"):
            w.write_model_group(
                "g",
                spec=spec,
                timestamps=np.zeros(3),
                lat=np.zeros(3),
                lng=np.zeros(3),
                # Wrong dim — spec says 4, this gives 8.
                embeddings=np.zeros((3, 8), dtype=np.float32),
            )


def test_writer_rejects_reserved_group_name(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    spec = schema.ModelGroupSpec(
        model="m", checkpoint="c", embedding_dim=4, sample_fps=1.0,
        sampling="s", preprocess="p", normalized=True,
    )
    with FeaturesWriter(out) as w:
        with pytest.raises(ValueError, match="reserved"):
            w.write_model_group(
                "gps",  # reserved for sensor groups
                spec=spec,
                timestamps=np.zeros(2),
                lat=np.zeros(2),
                lng=np.zeros(2),
                embeddings=np.zeros((2, 4), dtype=np.float32),
            )


def test_writer_rejects_double_write(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    ts = np.array([0.0, 1.0])
    lat = np.array([0.0, 0.0])
    lng = np.array([0.0, 0.0])
    with FeaturesWriter(out) as w:
        w.write_gps_group(timestamps=ts, lat=lat, lng=lng)
        with pytest.raises(RuntimeError, match="already"):
            w.write_gps_group(timestamps=ts, lat=lat, lng=lng)


def test_writer_close_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "f.h5"
    w = FeaturesWriter(out)
    w.close()
    w.close()  # second close must not raise.
