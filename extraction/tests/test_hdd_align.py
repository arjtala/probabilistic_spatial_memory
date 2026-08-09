"""Regression tests for the HDD video/GPS clock alignment.

These pin the two halves of the 2026-08 postmortem (see docs/HDD.md):
the tz-aware skew computation, and the coverage guard that makes an
``np.interp`` endpoint clamp loud instead of silent.
"""

from datetime import datetime

import numpy as np
import pytest

from psm_extraction.io.hdd import (
    HDDTrajectory,
    assert_track_coverage,
    video_start_skew,
)


def _traj(first_iso: str, t0: float = 1_507_322_704.0) -> HDDTrajectory:
    ts = t0 + np.arange(0.0, 100.0, 0.1)
    return HDDTrajectory(
        timestamps=ts,
        lat=np.full(ts.size, 37.4),
        lng=np.full(ts.size, -122.1),
        drive_id="201710061345",
        drive_day="2017_10_06_ITS1",
        bbox_extent_m=0.0,
        first_iso=first_iso,
    )


def _drive_dir(tmp_path, video_stem: str):
    center = tmp_path / "camera" / "center"
    center.mkdir(parents=True)
    (center / f"{video_stem}_center.mp4").write_bytes(b"")
    return tmp_path


def test_tz_aware_first_iso_yields_warmup_scale_skew(tmp_path) -> None:
    """A ``-08:00`` fix is already capture-local: skew is seconds, not hours.

    The original code called argless ``astimezone()``, which on a UTC host
    re-expressed the fix as UTC and returned offset + warmup (~28,803 s).
    """
    drive = _drive_dir(tmp_path, "2017-02-27-10-17-27")
    skew = video_start_skew(drive, _traj("2017-02-27T10:17:30.007385-08:00"))
    assert skew == pytest.approx(3.007385, abs=1e-3)


def test_naive_and_tz_aware_first_iso_agree(tmp_path) -> None:
    """The offset annotates the zone; it must not shift the wall clock."""
    drive = _drive_dir(tmp_path, "2017-02-27-10-17-27")
    naive = video_start_skew(drive, _traj("2017-02-27T10:17:30.007385"))
    aware = video_start_skew(drive, _traj("2017-02-27T10:17:30.007385-08:00"))
    assert naive == pytest.approx(aware, abs=1e-6)


def test_skew_is_zero_without_a_video(tmp_path) -> None:
    assert video_start_skew(tmp_path, _traj("2017-02-27T10:17:30-08:00")) == 0.0


def test_coverage_guard_allows_small_end_overhang() -> None:
    xp = 1_000.0 + np.arange(0.0, 100.0, 0.1)
    frame_abs = np.arange(999.0, 1_100.0, 1.0)  # ~1 s over each end
    assert_track_coverage(frame_abs, xp, drive_id="ok")


def test_coverage_guard_rejects_whole_track_offset() -> None:
    """The exact failure mode: frames displaced by a UTC offset."""
    xp = 1_507_322_704.0 + np.arange(0.0, 100.0, 0.1)
    frame_abs = xp - 25_204.0  # 7 h + warmup earlier
    with pytest.raises(ValueError, match="outside the RTK span"):
        assert_track_coverage(frame_abs, xp, drive_id="201710061345")


def test_coverage_guard_rejects_empty_inputs() -> None:
    xp = np.arange(10.0)
    with pytest.raises(ValueError, match="empty frame or RTK time axis"):
        assert_track_coverage(np.array([]), xp)


def test_clamped_interp_would_have_collapsed_the_track() -> None:
    """Documents *why* the guard exists, not just that it fires."""
    xp = 1_000.0 + np.arange(0.0, 100.0, 0.1)
    lat = 37.4 + np.linspace(0.0, 0.01, xp.size)
    out_of_range = xp - 25_204.0
    clamped = np.interp(out_of_range, xp, lat)
    assert np.unique(clamped).size == 1  # no exception, one coordinate
    with pytest.raises(ValueError):
        assert_track_coverage(out_of_range, xp)
