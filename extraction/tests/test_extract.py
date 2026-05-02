"""End-to-end orchestrator test using a stub runner — no ffmpeg, no torch."""

from pathlib import Path
from typing import Sequence
from unittest import mock

import h5py
import numpy as np
import pytest

from psm_extraction import schema
from psm_extraction.extract import ExtractOptions, extract
from psm_extraction.models import ModelRunner


class StubRunner(ModelRunner):
    def __init__(self) -> None:
        self.model_id = "stub/clip"
        self.checkpoint = "stub-fixed-1"
        self.embedding_dim = 4
        self.normalized = True
        self.preprocess = "stub"
        self.patch_grid = None
        self.backend = "stub"

    def embed_images(
        self,
        paths: Sequence[Path],
        batch_size: int = 16,
        *,
        progress=None,
    ) -> np.ndarray:
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((len(paths), self.embedding_dim)).astype(
            np.float32
        )
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        if progress is not None:
            progress(len(paths))
        return feats / np.clip(norms, 1e-12, None)

    def embed_text(self, query: str) -> np.ndarray:
        return np.ones(self.embedding_dim, dtype=np.float32) / np.sqrt(
            self.embedding_dim
        )


def _fake_extract_frames(video, fps, out_dir, *, verbose=False, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(6):
        p = out_dir / f"frame_{i:06d}.jpg"
        p.write_bytes(b"\xff\xd8\xff")  # tiny placeholder; runner is a stub
        paths.append(p)
    return paths


def _fake_video_duration(video, *, verbose=False):
    return 6.0


@pytest.fixture(autouse=True)
def _patch_io(monkeypatch):
    from psm_extraction import extract as extract_mod

    monkeypatch.setattr(extract_mod, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(extract_mod, "video_duration", _fake_video_duration)
    yield


def test_extract_writes_v2_compliant_file_with_synthetic_grid(tmp_path: Path) -> None:
    video = tmp_path / "data.mp4"
    video.write_bytes(b"")  # exists check only; runner is a stub
    output = tmp_path / "out.h5"

    runner = StubRunner()
    result = extract(
        ExtractOptions(
            video=video,
            output=output,
            runners=[("clip", runner)],
            sample_fps=2.0,
            segment_sec=1.0,
            use_gps=False,
        )
    )

    assert result.frame_count == 6
    assert result.track_mode == "synthetic_snake_grid"
    assert result.track_source is None
    assert result.group_names == ["clip"]
    assert result.embedding_dims == {"clip": 4}

    with h5py.File(output, "r") as h:
        assert int(h.attrs["schema_version"]) == schema.SCHEMA_VERSION
        assert "clip" in h
        group = h["clip"]
        assert group["embeddings"].shape == (6, 4)
        assert int(group.attrs["embedding_dim"]) == 4
        assert str(group.attrs["model"]) == "stub/clip"
        assert float(group.attrs["sample_fps"]) == 2.0
        assert bool(group.attrs["normalized"]) is True


def test_extract_writes_sensor_groups_from_sidecars_even_when_track_is_h5(
    tmp_path: Path,
) -> None:
    """Auto-detected gps.json + imu.json must populate sensor groups even
    when the per-frame track came from a sibling features.h5."""
    import json as _json

    # h5 sibling (track interpolation source) — has dino with timestamps/lat/lng.
    sibling = tmp_path / "features.h5"
    with h5py.File(sibling, "w") as h:
        dino = h.create_group("dino")
        dino.create_dataset("timestamps", data=np.linspace(1000.0, 1010.0, 11))
        dino.create_dataset("lat", data=np.full(11, 51.0))
        dino.create_dataset("lng", data=np.full(11, -0.18))
        dino.create_dataset("embeddings", data=np.zeros((11, 4), dtype=np.float32))

    # JSON sidecars in the same dir (sensor-group sources).
    (tmp_path / "gps.json").write_text(
        _json.dumps([{
            "stream_id": "281-1",
            "samples": [
                {"timestamp": 1.0, "latitude": 51.0, "longitude": -0.18},
                {"timestamp": 2.0, "latitude": 51.001, "longitude": -0.18},
            ],
        }])
    )
    (tmp_path / "imu.json").write_text(
        _json.dumps([{
            "stream_id": "1202-1",
            "samples": [
                {"timestamp": 0.001, "accel": [0.1, 0.0, 9.8], "gyro": [0.0, 0.0, 0.0]},
                {"timestamp": 0.002, "accel": [0.2, 0.0, 9.8], "gyro": [0.0, 0.0, 0.0]},
            ],
        }])
    )

    video = tmp_path / "data.mp4"
    video.write_bytes(b"")
    output = tmp_path / "out.h5"

    runner = StubRunner()
    result = extract(
        ExtractOptions(
            video=video,
            output=output,
            runners=[("clip", runner)],
            sample_fps=2.0,
            segment_sec=1.0,
            use_gps=True,
        )
    )

    assert result.track_mode == "real_gps"
    assert result.track_source == sibling
    assert set(result.sensor_groups_written) == {"gps", "imu"}

    with h5py.File(output, "r") as h:
        assert "gps" in h
        assert "imu" in h
        assert h["gps/timestamps"].shape == (2,)
        assert h["imu/accel"].shape == (2, 3)


def test_extract_uses_real_gps_when_available(tmp_path: Path) -> None:
    # Build a sibling features.h5 with a dino group covering the video range.
    sibling = tmp_path / "features.h5"
    with h5py.File(sibling, "w") as h:
        dino = h.create_group("dino")
        dino.create_dataset("timestamps", data=np.linspace(1000.0, 1010.0, 11))
        dino.create_dataset("lat", data=np.linspace(51.0, 51.1, 11))
        dino.create_dataset("lng", data=np.full(11, -0.18))
        dino.create_dataset(
            "embeddings", data=np.zeros((11, 4), dtype=np.float32)
        )

    video = tmp_path / "data.mp4"
    video.write_bytes(b"")
    output = tmp_path / "out.h5"

    runner = StubRunner()
    result = extract(
        ExtractOptions(
            video=video,
            output=output,
            runners=[("clip", runner)],
            sample_fps=2.0,
            segment_sec=1.0,
            use_gps=True,
        )
    )

    assert result.track_mode == "real_gps"
    assert result.track_source == sibling
    assert result.track_source_group == "dino"
    with h5py.File(output, "r") as h:
        # Frame timestamps span 0..2.5s; the track runs 0..10s with lat
        # linearly increasing from 51.0 to 51.1 (i.e. 0.01 / sec). Six
        # frames at 2 fps therefore land on lat = 51.0 + t*0.01.
        frame_t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        np.testing.assert_allclose(h["clip/lat"][...], 51.0 + frame_t * 0.01)
        assert "interpolation" in h["clip"].attrs


def test_extract_writes_multiple_model_groups(tmp_path: Path) -> None:
    """Two stub runners → two model groups in one file, one frame pass."""
    video = tmp_path / "data.mp4"
    video.write_bytes(b"")

    class StubDino(StubRunner):
        def __init__(self) -> None:
            super().__init__()
            self.model_id = "stub/dinov2"
            self.checkpoint = "stub-dino-1"
            self.embedding_dim = 8
            self.patch_grid = (2, 2)
            self.preprocess = "stub-dino"

        def embed_images(
            self,
            paths: Sequence[Path],
            batch_size: int = 16,
            *,
            progress=None,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            embeddings = np.zeros((len(paths), self.embedding_dim), dtype=np.float32)
            attention = np.zeros((len(paths), 2, 2), dtype=np.float32)
            return embeddings, attention

    runners: list[tuple[str, ModelRunner]] = [
        ("clip", StubRunner()),
        ("dino", StubDino()),
    ]
    result = extract(
        ExtractOptions(
            video=video,
            output=tmp_path / "out.h5",
            runners=runners,
            sample_fps=2.0,
            segment_sec=1.0,
            use_gps=False,
        )
    )
    assert result.group_names == ["clip", "dino"]
    assert result.embedding_dims == {"clip": 4, "dino": 8}

    with h5py.File(result.features_path, "r") as h:
        assert h["clip/embeddings"].shape == (6, 4)
        assert h["dino/embeddings"].shape == (6, 8)
        assert h["dino/attention_maps"].shape == (6, 2, 2)
        assert list(h["dino"].attrs["patch_grid"]) == [2, 2]


def test_extract_rejects_reserved_group(tmp_path: Path) -> None:
    video = tmp_path / "data.mp4"
    video.write_bytes(b"")
    runner = StubRunner()
    with pytest.raises(ValueError, match="reserved"):
        extract(
            ExtractOptions(
                video=video,
                output=tmp_path / "out.h5",
                runners=[("gps", runner)],
            )
        )


def test_extract_rejects_duplicate_group_names(tmp_path: Path) -> None:
    video = tmp_path / "data.mp4"
    video.write_bytes(b"")
    with pytest.raises(ValueError, match="duplicate"):
        extract(
            ExtractOptions(
                video=video,
                output=tmp_path / "out.h5",
                runners=[("clip", StubRunner()), ("clip", StubRunner())],
                use_gps=False,
            )
        )


def test_extract_rejects_runner_dim_mismatch(tmp_path: Path) -> None:
    video = tmp_path / "data.mp4"
    video.write_bytes(b"")

    class WrongDimRunner(StubRunner):
        def embed_images(self, paths, batch_size: int = 16, *, progress=None):
            return np.zeros((len(paths), self.embedding_dim + 1), dtype=np.float32)

    runner = WrongDimRunner()
    with pytest.raises(RuntimeError, match="embedding_dim"):
        extract(
            ExtractOptions(
                video=video,
                output=tmp_path / "out.h5",
                runners=[("clip", runner)],
                use_gps=False,
            )
        )
