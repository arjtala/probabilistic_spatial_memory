"""Migration tests: synthesize a v1-shaped file, migrate, audit attrs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from psm_extraction import schema
from psm_extraction.migrate import migrate_v1_to_v2


def _write_v1_fixture(path: Path, *, with_jepa: bool = True) -> None:
    """Mimic the existing external pipeline's output: data only, no attrs."""
    with h5py.File(path, "w") as h:
        # gps (sensor group, untouched by migrate)
        gps = h.create_group("gps")
        gps.create_dataset("timestamps", data=np.linspace(0.0, 10.0, 11))
        gps.create_dataset("lat", data=np.full(11, 51.46))
        gps.create_dataset("lng", data=np.full(11, -0.18))

        # imu (sensor group, untouched)
        imu = h.create_group("imu")
        imu.create_dataset("timestamps", data=np.linspace(0.0, 10.0, 1001))
        imu.create_dataset("accel", data=np.zeros((1001, 3), dtype=np.float32))
        imu.create_dataset("gyro", data=np.zeros((1001, 3), dtype=np.float32))

        # dino model group
        n = 300
        dino = h.create_group("dino")
        dino.create_dataset("timestamps", data=np.linspace(0.0, 10.0, n))
        dino.create_dataset("lat", data=np.full(n, 51.46))
        dino.create_dataset("lng", data=np.full(n, -0.18))
        dino.create_dataset(
            "embeddings", data=np.zeros((n, 1024), dtype=np.float32)
        )
        dino.create_dataset(
            "attention_maps", data=np.zeros((n, 14, 14), dtype=np.float32)
        )

        if with_jepa:
            m = 40
            jepa = h.create_group("jepa")
            jepa.create_dataset("timestamps", data=np.linspace(0.0, 10.0, m))
            jepa.create_dataset("lat", data=np.full(m, 51.46))
            jepa.create_dataset("lng", data=np.full(m, -0.18))
            jepa.create_dataset(
                "embeddings", data=np.zeros((m, 1024), dtype=np.float32)
            )
            jepa.create_dataset(
                "prediction_maps",
                data=np.zeros((m, 16, 16), dtype=np.float32),
            )


def test_migrate_v1_adds_root_attrs(tmp_path: Path) -> None:
    fixture = tmp_path / "features.h5"
    _write_v1_fixture(fixture)

    report = migrate_v1_to_v2(fixture)
    assert "root_added" in report
    for required in schema.ROOT_REQUIRED_ATTRS:
        assert required in report["root_added"]

    with h5py.File(fixture, "r") as h:
        assert int(h.attrs["schema_version"]) == schema.SCHEMA_VERSION
        assert str(h.attrs["producer"]) == schema.PRODUCER_NAME
        assert "created_at_utc" in h.attrs


def test_migrate_v1_fills_known_group_attrs(tmp_path: Path) -> None:
    fixture = tmp_path / "features.h5"
    _write_v1_fixture(fixture)

    report = migrate_v1_to_v2(fixture)
    assert "dino" in report["groups"]
    added = set(report["groups"]["dino"]["added"])
    for required in schema.MODEL_REQUIRED_ATTRS:
        assert required in added

    with h5py.File(fixture, "r") as h:
        dino = h["dino"]
        assert int(dino.attrs["embedding_dim"]) == 1024
        assert float(dino.attrs["sample_fps"]) > 0.0
        assert list(dino.attrs["patch_grid"]) == [14, 14]
        assert str(dino.attrs["model"]).startswith("facebook")


def test_migrate_v1_skips_sensor_groups(tmp_path: Path) -> None:
    fixture = tmp_path / "features.h5"
    _write_v1_fixture(fixture)

    migrate_v1_to_v2(fixture)
    with h5py.File(fixture, "r") as h:
        # gps and imu must not have model-group attrs after migration.
        assert "model" not in h["gps"].attrs
        assert "embedding_dim" not in h["imu"].attrs


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    fixture = tmp_path / "features.h5"
    _write_v1_fixture(fixture)

    first = migrate_v1_to_v2(fixture)
    assert "root_added" in first

    second = migrate_v1_to_v2(fixture)
    assert second.get("already_at") == schema.SCHEMA_VERSION


def test_migrate_passes_through_extra_root_attrs(tmp_path: Path) -> None:
    fixture = tmp_path / "features.h5"
    _write_v1_fixture(fixture)

    migrate_v1_to_v2(
        fixture,
        extra_root_attrs={"source_video": "/path/to/data.mp4", "session_id": "xyz"},
    )
    with h5py.File(fixture, "r") as h:
        assert str(h.attrs["source_video"]) == "/path/to/data.mp4"
        assert str(h.attrs["session_id"]) == "xyz"


def test_migrate_skips_unknown_groups(tmp_path: Path) -> None:
    fixture = tmp_path / "features.h5"
    with h5py.File(fixture, "w") as h:
        unknown = h.create_group("future_model")
        unknown.create_dataset("timestamps", data=np.zeros(2))
        unknown.create_dataset("lat", data=np.zeros(2))
        unknown.create_dataset("lng", data=np.zeros(2))
        unknown.create_dataset("embeddings", data=np.zeros((2, 8)))

    report = migrate_v1_to_v2(fixture)
    assert report["groups"]["future_model"] == {"skipped": "unknown_group"}
    with h5py.File(fixture, "r") as h:
        assert "model" not in h["future_model"].attrs
        assert int(h.attrs["schema_version"]) == schema.SCHEMA_VERSION


def test_migrate_writer_produced_file_is_noop(tmp_path: Path) -> None:
    """A file written by FeaturesWriter is already v2; migrate must no-op."""
    from psm_extraction.writer import FeaturesWriter

    fixture = tmp_path / "fresh.h5"
    spec = schema.ModelGroupSpec(
        model="m", checkpoint="c", embedding_dim=4, sample_fps=1.0,
        sampling="s", preprocess="p", normalized=True,
    )
    with FeaturesWriter(fixture) as w:
        w.write_model_group(
            "dino",
            spec=spec,
            timestamps=np.zeros(2),
            lat=np.zeros(2),
            lng=np.zeros(2),
            embeddings=np.zeros((2, 4), dtype=np.float32),
        )
    report = migrate_v1_to_v2(fixture)
    assert report.get("already_at") == schema.SCHEMA_VERSION
