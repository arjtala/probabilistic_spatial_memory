"""Tests for the Aria JSON sidecar reader (gps.json + imu.json)."""

import json
from pathlib import Path

import numpy as np
import pytest

from psm_extraction.io.json_sidecar import (
    capture_time_epoch,
    read_gps_json,
    read_imu_json,
    read_metadata_json,
)


def _write_gps_fixture(path: Path, samples: list[dict], stream_id: str = "281-1") -> None:
    payload = [{"stream_id": stream_id, "samples": samples}]
    path.write_text(json.dumps(payload))


def _write_imu_fixture(path: Path, samples: list[dict], stream_id: str = "1202-1") -> None:
    payload = [{"stream_id": stream_id, "samples": samples}]
    path.write_text(json.dumps(payload))


def test_read_gps_json_filters_pre_fix_samples(tmp_path: Path) -> None:
    """Pre-fix samples (denormal timestamp + zero lat/lng) must be skipped."""
    fixture = tmp_path / "gps.json"
    _write_gps_fixture(
        fixture,
        [
            # Two pre-fix samples that should be filtered out.
            {"timestamp": 5e-324, "latitude": 0.0, "longitude": 0.0},
            {"timestamp": 5e-324, "latitude": 0.0, "longitude": 0.0},
            # Two real fixes.
            {"timestamp": 65.5, "latitude": 51.467, "longitude": -0.187},
            {"timestamp": 66.5, "latitude": 51.468, "longitude": -0.186},
        ],
    )
    sidecar = read_gps_json(fixture)
    np.testing.assert_array_equal(sidecar.timestamps, [65.5, 66.5])
    np.testing.assert_array_equal(sidecar.lat, [51.467, 51.468])
    assert sidecar.stream_id == "281-1"


def test_read_gps_json_picks_largest_stream(tmp_path: Path) -> None:
    fixture = tmp_path / "gps.json"
    payload = [
        {
            "stream_id": "281-1",
            "samples": [
                {"timestamp": 1.0, "latitude": 51.4, "longitude": -0.1},
                {"timestamp": 2.0, "latitude": 51.5, "longitude": -0.1},
            ],
        },
        {
            "stream_id": "281-2",
            "samples": [
                {"timestamp": 1.0, "latitude": 51.0, "longitude": 0.0},
            ],
        },
    ]
    fixture.write_text(json.dumps(payload))
    sidecar = read_gps_json(fixture)
    assert sidecar.stream_id == "281-1"
    assert sidecar.timestamps.shape[0] == 2


def test_read_gps_json_explicit_stream_id(tmp_path: Path) -> None:
    fixture = tmp_path / "gps.json"
    payload = [
        {
            "stream_id": "281-1",
            "samples": [{"timestamp": 1.0, "latitude": 51.4, "longitude": -0.1}],
        },
        {
            "stream_id": "281-2",
            "samples": [{"timestamp": 2.0, "latitude": 51.5, "longitude": -0.2}],
        },
    ]
    fixture.write_text(json.dumps(payload))
    sidecar = read_gps_json(fixture, stream_id="281-2")
    assert sidecar.stream_id == "281-2"
    np.testing.assert_array_equal(sidecar.lat, [51.5])


def test_read_gps_json_raises_when_no_valid_samples(tmp_path: Path) -> None:
    fixture = tmp_path / "gps.json"
    _write_gps_fixture(fixture, [{"timestamp": 5e-324, "latitude": 0.0, "longitude": 0.0}])
    with pytest.raises(RuntimeError, match="no valid GPS"):
        read_gps_json(fixture)


def test_read_gps_json_sorts_by_timestamp(tmp_path: Path) -> None:
    fixture = tmp_path / "gps.json"
    _write_gps_fixture(
        fixture,
        [
            {"timestamp": 100.0, "latitude": 51.0, "longitude": -0.1},
            {"timestamp": 50.0, "latitude": 51.5, "longitude": -0.2},
            {"timestamp": 75.0, "latitude": 51.25, "longitude": -0.15},
        ],
    )
    sidecar = read_gps_json(fixture)
    np.testing.assert_array_equal(sidecar.timestamps, [50.0, 75.0, 100.0])
    np.testing.assert_array_equal(sidecar.lat, [51.5, 51.25, 51.0])


def test_read_imu_json_round_trips(tmp_path: Path) -> None:
    fixture = tmp_path / "imu.json"
    _write_imu_fixture(
        fixture,
        [
            {"timestamp": 0.001, "accel": [0.1, 0.2, 9.8], "gyro": [0.01, 0.02, 0.03]},
            {"timestamp": 0.002, "accel": [0.15, 0.25, 9.85], "gyro": [0.0, 0.0, 0.0]},
        ],
    )
    sidecar = read_imu_json(fixture)
    assert sidecar.stream_id == "1202-1"
    assert sidecar.timestamps.shape == (2,)
    assert sidecar.accel.shape == (2, 3)
    assert sidecar.gyro.shape == (2, 3)
    assert sidecar.accel.dtype == np.float32
    np.testing.assert_allclose(sidecar.accel[0], [0.1, 0.2, 9.8], rtol=1e-5)


def test_read_imu_json_skips_invalid_samples(tmp_path: Path) -> None:
    fixture = tmp_path / "imu.json"
    _write_imu_fixture(
        fixture,
        [
            {"timestamp": 0.001, "accel": [0.1, 0.2, 9.8], "gyro": [0.01, 0.02, 0.03]},
            {"timestamp": 0.002, "accel": [float("nan"), 0.25, 9.85], "gyro": [0.0, 0.0, 0.0]},
            {"timestamp": 0.003, "accel": [0.2, 0.3, 9.9], "gyro": [0.0, 0.0, 0.0]},
        ],
    )
    sidecar = read_imu_json(fixture)
    assert sidecar.timestamps.shape[0] == 2  # NaN sample dropped


def test_read_metadata_json_and_capture_time(tmp_path: Path) -> None:
    fixture = tmp_path / "metadata.json"
    fixture.write_text(json.dumps({
        "recording_id": "abc",
        "tags": {"capture_time_epoch": "1678365188"},
    }))
    meta = read_metadata_json(fixture)
    assert meta["recording_id"] == "abc"
    assert capture_time_epoch(meta) == 1678365188.0


def test_capture_time_epoch_returns_none_when_absent() -> None:
    assert capture_time_epoch({}) is None
    assert capture_time_epoch({"tags": {"capture_time_epoch": "not-a-number"}}) is None
