"""Pure-numpy/h5py tests for align.py — no torch / transformers / mlx."""

from pathlib import Path

import h5py
import numpy as np

from psm_extraction.align import (
    SessionTrack,
    load_session_track,
    map_frames_to_gps,
    synthetic_snake_grid,
)


def test_synthetic_snake_grid_groups_frames_by_segment() -> None:
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    lats, lngs, count = synthetic_snake_grid(
        timestamps, segment_sec=1.0, grid_columns=4, cell_step_deg=0.01
    )
    assert lats[0] == lats[1]
    assert lngs[0] == lngs[1]
    assert (lats[2], lngs[2]) != (lats[0], lngs[0])
    assert count >= 2


def test_synthetic_snake_grid_snake_pattern_alternates_columns() -> None:
    # 9 segments at columns=3 → row 0 fills cols 0,1,2; row 1 reverses to 2,1,0;
    # so segment 0 and segment 5 (row 1, col 0 after reversal) share a longitude.
    timestamps = np.arange(9.0)
    lats, lngs, _ = synthetic_snake_grid(
        timestamps, segment_sec=1.0, grid_columns=3, cell_step_deg=0.01
    )
    assert lngs[0] == lngs[5]
    assert lats[0] != lats[5]


def test_synthetic_snake_grid_rejects_nonpositive_segment() -> None:
    import pytest

    with pytest.raises(ValueError):
        synthetic_snake_grid(np.zeros(3), segment_sec=0.0)


def test_map_frames_to_gps_linear_interpolates() -> None:
    track = SessionTrack(
        rel_seconds=np.array([0.0, 10.0]),
        lat=np.array([0.0, 1.0]),
        lng=np.array([0.0, 0.0]),
        source_group="gps",
    )
    frames = np.array([0.0, 5.0, 10.0])
    lats, lngs = map_frames_to_gps(frames, track)
    np.testing.assert_allclose(lats, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(lngs, [0.0, 0.0, 0.0])


def test_map_frames_to_gps_clips_to_track_range() -> None:
    track = SessionTrack(
        rel_seconds=np.array([5.0, 15.0]),
        lat=np.array([10.0, 20.0]),
        lng=np.array([0.0, 0.0]),
        source_group="gps",
    )
    # Frames before / after the track range should clamp to the endpoints
    # rather than extrapolate (otherwise a single short GPS gap would warp
    # large stretches of the video).
    frames = np.array([0.0, 5.0, 15.0, 30.0])
    lats, lngs = map_frames_to_gps(frames, track)
    np.testing.assert_allclose(lats, [10.0, 10.0, 20.0, 20.0])


def test_load_session_track_prefers_dino_over_gps(tmp_path: Path) -> None:
    p = tmp_path / "f.h5"
    with h5py.File(p, "w") as h:
        gps = h.create_group("gps")
        gps.create_dataset("timestamps", data=np.array([1000.0, 1010.0]))
        gps.create_dataset("lat", data=np.array([5.0, 6.0]))
        gps.create_dataset("lng", data=np.array([0.0, 0.0]))
        dino = h.create_group("dino")
        dino.create_dataset("timestamps", data=np.linspace(1000.0, 1010.0, 11))
        dino.create_dataset("lat", data=np.full(11, 7.0))
        dino.create_dataset("lng", data=np.full(11, 0.0))
    track = load_session_track(p)
    assert track is not None
    assert track.source_group == "dino"
    assert track.lat[0] == 7.0
    assert track.rel_seconds[0] == 0.0


def test_load_session_track_skips_nonfinite_samples(tmp_path: Path) -> None:
    p = tmp_path / "f.h5"
    with h5py.File(p, "w") as h:
        gps = h.create_group("gps")
        gps.create_dataset(
            "timestamps", data=np.array([0.0, 1.0, 2.0, 3.0])
        )
        gps.create_dataset("lat", data=np.array([0.0, np.nan, 2.0, 3.0]))
        gps.create_dataset("lng", data=np.array([0.0, 0.0, 0.0, 0.0]))
    track = load_session_track(p)
    assert track is not None
    # The NaN sample at index 1 must have been dropped.
    assert track.rel_seconds.shape[0] == 3
    np.testing.assert_array_equal(track.lat, np.array([0.0, 2.0, 3.0]))


def test_load_session_track_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_session_track(tmp_path / "nope.h5") is None


def test_load_session_track_returns_none_for_unrelated_h5(tmp_path: Path) -> None:
    p = tmp_path / "f.h5"
    with h5py.File(p, "w") as h:
        h.create_dataset("data", data=np.zeros(10))
    assert load_session_track(p) is None
