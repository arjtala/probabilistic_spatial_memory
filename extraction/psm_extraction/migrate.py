"""In-place migration of schema-v1 features.h5 to schema-v2.

Adds missing root + per-group attrs using best-effort defaults for known
group names (`dino`, `jepa`, `clip`). Sensor groups (`gps`, `imu`) are left
alone — only an optional `rate_hz_nominal` would belong there and we can't
infer it without the source pipeline metadata.

Datasets are never modified. The migration is idempotent: running it on an
already-v2 file is a no-op that returns `{"already_at": SCHEMA_VERSION}`.
"""

import datetime
from pathlib import Path
from typing import Any

import h5py

from . import schema

# Best-effort attr defaults per known model group. These match what the
# external Aria extraction pipeline appears to produce (see existing
# features.h5 dumps). The `checkpoint` is a placeholder — a v1 file simply
# doesn't record which one was used; we mark that explicitly so audits
# don't conflate "back-filled default" with "ground truth".
KNOWN_GROUP_DEFAULTS: dict[str, dict[str, Any]] = {
    "dino": {
        "model": "facebookresearch/dinov3",
        "checkpoint": "unknown_v1",
        "patch_grid": [14, 14],
        "preprocess": "resize=224,center_crop=224,normalize=imagenet",
        "normalized": True,
        "sampling": "every_video_frame",
        "interpolation": "linear,nearest_at_edges,from=imu+gps",
    },
    "jepa": {
        "model": "facebookresearch/v-jepa-2",
        "checkpoint": "unknown_v1",
        "patch_grid": [16, 16],
        "preprocess": "resize=224,center_crop=224",
        "normalized": False,
        "sampling": "downsampled_from_video",
        "interpolation": "linear,nearest_at_edges,from=imu+gps",
    },
    "clip": {
        "model": "openai/clip",
        "checkpoint": "unknown_v1",
        "preprocess": "clip_default",
        "normalized": True,
        "sampling": "downsampled_from_video",
    },
}


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _embedding_dim(group: h5py.Group) -> int | None:
    if "embeddings" not in group:
        return None
    embeddings = group["embeddings"]
    if not isinstance(embeddings, h5py.Dataset) or embeddings.ndim != 2:
        return None
    return int(embeddings.shape[1])


def _sample_fps(group: h5py.Group) -> float | None:
    if "timestamps" not in group:
        return None
    ts = group["timestamps"]
    if not isinstance(ts, h5py.Dataset) or ts.ndim != 1 or ts.shape[0] < 2:
        return None
    arr = ts[:]
    span = float(arr[-1] - arr[0])
    if span <= 0.0:
        return None
    return float((arr.shape[0] - 1) / span)


def migrate_v1_to_v2(
    path: Path | str,
    *,
    producer_version: str = "0.1.0",
    extra_root_attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add v2 attrs in place to an existing features.h5 file.

    Returns a report dict listing exactly which attrs were back-filled and
    which were left intact. Use `extra_root_attrs` to pass `source_video` /
    `session_id` if the caller knows them; otherwise they're left absent.
    """
    path = Path(path)
    report: dict[str, Any] = {"path": str(path), "root_added": [], "groups": {}}

    with h5py.File(path, "a") as h:
        existing_version = h.attrs.get("schema_version")
        if existing_version is not None and int(existing_version) >= schema.SCHEMA_VERSION:
            return {"already_at": int(existing_version), "path": str(path)}

        root_defaults = {
            "schema_version": schema.SCHEMA_VERSION,
            "producer": schema.PRODUCER_NAME,
            "producer_version": producer_version,
            "created_at_utc": _now_utc_iso(),
            "timestamp_unit": schema.DEFAULT_TIMESTAMP_UNIT,
            "coord_system": schema.DEFAULT_COORD_SYSTEM,
        }
        if extra_root_attrs:
            for k, v in extra_root_attrs.items():
                if k in {"schema_version", "producer"}:
                    raise ValueError(f"refusing to override reserved root attr {k!r}")
                root_defaults.setdefault(k, v)

        for key, value in root_defaults.items():
            if key not in h.attrs:
                h.attrs[key] = value
                report["root_added"].append(key)

        for name, item in h.items():
            if not isinstance(item, h5py.Group):
                continue
            if name in {"gps", "imu"}:
                continue
            defaults = KNOWN_GROUP_DEFAULTS.get(name)
            if defaults is None:
                report["groups"][name] = {"skipped": "unknown_group"}
                continue
            added: list[str] = []
            inferred_dim = _embedding_dim(item)
            inferred_fps = _sample_fps(item)
            for attr_key, attr_val in defaults.items():
                if attr_key not in item.attrs:
                    item.attrs[attr_key] = attr_val
                    added.append(attr_key)
            if "embedding_dim" not in item.attrs and inferred_dim is not None:
                item.attrs["embedding_dim"] = inferred_dim
                added.append("embedding_dim")
            if "sample_fps" not in item.attrs and inferred_fps is not None:
                item.attrs["sample_fps"] = inferred_fps
                added.append("sample_fps")
            report["groups"][name] = {"added": added}

    return report
