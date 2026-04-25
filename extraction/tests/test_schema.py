"""Schema constants and ModelGroupSpec basics."""

import pytest

from psm_extraction import schema


def test_schema_version_is_two() -> None:
    assert schema.SCHEMA_VERSION == 2


def test_required_root_attrs_include_metadata() -> None:
    assert "schema_version" in schema.ROOT_REQUIRED_ATTRS
    assert "producer" in schema.ROOT_REQUIRED_ATTRS
    assert "timestamp_unit" in schema.ROOT_REQUIRED_ATTRS
    assert "coord_system" in schema.ROOT_REQUIRED_ATTRS


def test_required_model_datasets_match_c_ingest_contract() -> None:
    # The C ingest reads timestamps, lat, lng, embeddings — those must stay
    # required to keep the C engine compatible with v2 files.
    assert set(schema.MODEL_REQUIRED_DATASETS) == {
        "timestamps",
        "lat",
        "lng",
        "embeddings",
    }


def test_model_group_spec_is_frozen() -> None:
    spec = schema.ModelGroupSpec(
        model="x", checkpoint="y", embedding_dim=8, sample_fps=1.0,
        sampling="z", preprocess="w", normalized=True,
    )
    with pytest.raises(dataclasses_FrozenInstanceError()):
        spec.model = "other"  # type: ignore[misc]


def dataclasses_FrozenInstanceError():
    # tiny shim so the test doesn't import dataclasses itself just for the
    # exception type at module top level.
    import dataclasses

    return dataclasses.FrozenInstanceError


def test_model_group_spec_optional_fields_default_none() -> None:
    spec = schema.ModelGroupSpec(
        model="x", checkpoint="y", embedding_dim=8, sample_fps=1.0,
        sampling="z", preprocess="w", normalized=True,
    )
    assert spec.patch_grid is None
    assert spec.interpolation is None
