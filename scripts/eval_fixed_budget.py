#!/usr/bin/env python3
"""Compare retention policies at an identical exemplar budget.

This is an allocation-policy experiment, not a replacement PSM engine. It
selects exactly M frame embeddings, ranks the retained bank globally by cosine,
and scores with the repository's standard look-back harness. Keeping query
ranking identical isolates the effect of retention policy.

Policies:
  global_reservoir  Algorithm R over the entire stream.
  fifo              The M most recent frames.
  hybrid            M/2 prefix-reservoir plus M/2 recent frames.
  uniform_time      M evenly spaced frames (offline baseline).
  semantic_kcenter  Offline cosine k-center diversity baseline.
  spatial_priority  Causal exact-M balancing with stable cell priorities.
  spatial_balanced  Water-fill M slots across H3 cells, reservoir per cell.
  visit_balanced    Water-fill across cells and then visit episodes.

Global reservoir, FIFO, hybrid, and spatial-priority have causal streaming
implementations. Uniform-time and the water-filled selectors use the completed
sequence; the water-filled results are non-causal diagnostics and must not be
described as online algorithms.
Spatial methods have an exact global *exemplar* budget. Their HLL register cost
is reported separately and still grows with the number of visited cells.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Hashable, TypeVar

import h5py
import numpy as np

from _eval_common import (
    embed_query_text,
    load_features,
    load_questions,
    score_predictions,
    summarize_question,
    write_eval_json,
)
from eval_brute_force_clip import topk_brute_force
from eval_global_reservoir import TIMESTAMP_BYTES, reservoir_indices


METHODS = (
    "global_reservoir",
    "fifo",
    "hybrid",
    "uniform_time",
    "semantic_kcenter",
    "spatial_priority",
    "spatial_balanced",
    "visit_balanced",
)
SPATIAL_METHODS = {"spatial_priority", "spatial_balanced", "visit_balanced"}
OFFLINE_SPATIAL_METHODS = {"spatial_balanced", "visit_balanced"}
EARTH_RADIUS_M = 6_371_008.8
GroupKey = TypeVar("GroupKey", bound=Hashable)


def parse_int_list(text: str, name: str, *, minimum: int = 1) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad {name}: {exc}") from exc
    if not values or any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(
            f"{name} must contain integers >= {minimum}"
        )
    return values


def parse_methods(text: str) -> list[str]:
    methods = [method.strip() for method in text.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if not methods or unknown:
        raise argparse.ArgumentTypeError(
            f"methods must be drawn from {','.join(METHODS)}; unknown={unknown}"
        )
    return list(dict.fromkeys(methods))


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{label}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_code_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/eval_fixed_budget.py",
        "scripts/eval_global_reservoir.py",
        "scripts/eval_brute_force_clip.py",
        "scripts/_eval_common.py",
        "extraction/psm_extraction/models/base.py",
        "extraction/psm_extraction/models/clip_pytorch.py",
        "extraction/psm_extraction/models/siglip_pytorch.py",
        "extraction/psm_extraction/models/registry.py",
    ):
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def waterfill_quotas(
    sizes: Mapping[GroupKey, int], budget: int, seed: int
) -> dict[GroupKey, int]:
    """Allocate exactly ``min(budget, sum(sizes))`` slots as evenly as possible."""
    if budget < 0 or any(size < 0 for size in sizes.values()):
        raise ValueError("budget and group sizes must be non-negative")
    quotas = {key: 0 for key in sizes}
    remaining = min(budget, sum(sizes.values()))
    if remaining == 0:
        return quotas

    keys = sorted(sizes, key=str)
    rng = np.random.default_rng(seed)
    order = [keys[int(i)] for i in rng.permutation(len(keys))]
    while remaining:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if quotas[key] < sizes[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("water filling stalled before exhausting budget")
    return quotas


def group_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        groups.setdefault(str(label), []).append(idx)
    return {
        label: np.asarray(indices, dtype=np.int64)
        for label, indices in groups.items()
    }


def visit_episodes(indices: np.ndarray, timestamps: np.ndarray, gap_sec: float) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    ordered = indices[np.argsort(timestamps[indices], kind="stable")]
    split_at = np.flatnonzero(np.diff(timestamps[ordered]) > gap_sec) + 1
    return [chunk for chunk in np.split(ordered, split_at) if chunk.size]


def select_spatial_balanced(
    groups: dict[str, np.ndarray], budget: int, seed: int
) -> np.ndarray:
    quotas = waterfill_quotas(
        {label: int(indices.size) for label, indices in groups.items()},
        budget,
        seed,
    )
    selected: list[np.ndarray] = []
    for label in sorted(groups):
        indices = groups[label]
        quota = quotas[label]
        local = reservoir_indices(
            int(indices.size), quota, stable_seed(seed, f"cell:{label}")
        )
        selected.append(indices[local])
    result = np.sort(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    if result.size != min(budget, sum(len(v) for v in groups.values())):
        raise RuntimeError("spatial selector did not fill its exact budget")
    return result


def semantic_kcenter_order(
    embeddings: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    """Return a farthest-first cosine k-center ordering up to ``budget``."""
    n_frames = int(embeddings.shape[0])
    budget = min(budget, n_frames)
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n_frames))
    selected = [first]
    chosen = np.zeros(n_frames, dtype=bool)
    chosen[first] = True
    nearest_similarity = embeddings @ embeddings[first]
    nearest_similarity[first] = np.inf
    while len(selected) < budget:
        idx = int(np.argmin(nearest_similarity))
        selected.append(idx)
        chosen[idx] = True
        nearest_similarity = np.maximum(
            nearest_similarity, embeddings @ embeddings[idx]
        )
        nearest_similarity[chosen] = np.inf
    return np.asarray(selected, dtype=np.int64)


def select_spatial_priority(labels: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Causally maintain exactly M exemplars while balancing observed cells.

    A frame from an underrepresented cell evicts a uniformly chosen exemplar
    from a currently most-represented cell. Once its cell is no longer
    underrepresented, ordinary Algorithm-R replacement applies within that
    cell. When cells outnumber slots, stable random cell priorities retain a
    uniform priority sample of M cells instead of a recent-cell cache. Quota
    expansion cannot recover previously discarded observations, so this is a
    deliberately simple online heuristic rather than uniform stratified
    sampling.
    """
    budget = min(budget, int(labels.size))
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    banks: dict[str, list[int]] = {}
    seen: dict[str, int] = {}
    retained = 0
    for stream_idx, raw_label in enumerate(labels.tolist()):
        label = str(raw_label)
        bank = banks.setdefault(label, [])
        seen[label] = seen.get(label, 0) + 1
        if retained < budget:
            bank.append(stream_idx)
            retained += 1
            continue

        target_size = len(bank)
        max_size = max(len(items) for items in banks.values())
        donors = [
            donor
            for donor, items in banks.items()
            if len(items) == max_size and len(items) > target_size
        ]
        if donors:
            if target_size == 0 and max_size == 1:
                donor = max(
                    donors,
                    key=lambda cell: stable_seed(seed, f"cell-priority:{cell}"),
                )
                target_priority = stable_seed(seed, f"cell-priority:{label}")
                donor_priority = stable_seed(seed, f"cell-priority:{donor}")
                if target_priority >= donor_priority:
                    banks.pop(label, None)
                    continue
            else:
                donor = donors[int(rng.integers(0, len(donors)))]
            donor_bank = banks[donor]
            donor_bank.pop(int(rng.integers(0, len(donor_bank))))
            if not donor_bank:
                del banks[donor]
            bank.append(stream_idx)
            continue

        if target_size:
            slot = int(rng.integers(0, seen[label]))
            if slot < target_size:
                bank[slot] = stream_idx

    result = np.sort(
        np.asarray(
            [idx for items in banks.values() for idx in items], dtype=np.int64
        )
    )
    if result.size != budget:
        raise RuntimeError("causal spatial selector did not preserve its exact budget")
    return result


def select_visit_balanced(
    groups: dict[str, np.ndarray],
    timestamps: np.ndarray,
    budget: int,
    seed: int,
    gap_sec: float,
) -> np.ndarray:
    cell_quotas = waterfill_quotas(
        {label: int(indices.size) for label, indices in groups.items()},
        budget,
        seed,
    )
    selected: list[np.ndarray] = []
    for label in sorted(groups):
        episodes = visit_episodes(groups[label], timestamps, gap_sec)
        episode_sizes = {idx: int(episode.size) for idx, episode in enumerate(episodes)}
        episode_quotas = waterfill_quotas(
            episode_sizes,
            cell_quotas[label],
            stable_seed(seed, f"episodes:{label}"),
        )
        for episode_idx, episode in enumerate(episodes):
            local = reservoir_indices(
                int(episode.size),
                episode_quotas[episode_idx],
                stable_seed(seed, f"cell:{label}:visit:{episode_idx}"),
            )
            selected.append(episode[local])
    result = np.sort(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    if result.size != min(budget, sum(len(v) for v in groups.values())):
        raise RuntimeError("visit selector did not fill its exact budget")
    return result


def select_indices(
    method: str,
    n_frames: int,
    budget: int,
    seed: int,
    labels: np.ndarray,
    groups: dict[str, np.ndarray],
    timestamps: np.ndarray,
    embeddings: np.ndarray,
    semantic_order: np.ndarray | None,
    visit_gap_sec: float,
) -> np.ndarray:
    budget = min(budget, n_frames)
    if method == "global_reservoir":
        return reservoir_indices(n_frames, budget, seed)
    if method == "fifo":
        return np.arange(n_frames - budget, n_frames, dtype=np.int64)
    if method == "hybrid":
        recent_count = budget // 2
        lifetime_count = budget - recent_count
        prefix_count = n_frames - recent_count
        lifetime = reservoir_indices(prefix_count, lifetime_count, seed)
        recent = np.arange(prefix_count, n_frames, dtype=np.int64)
        return np.sort(np.concatenate((lifetime, recent)))
    if method == "uniform_time":
        if budget == 0:
            return np.empty(0, dtype=np.int64)
        return np.rint(np.linspace(0, n_frames - 1, budget)).astype(np.int64)
    if method == "semantic_kcenter":
        order = (
            semantic_order
            if semantic_order is not None
            else semantic_kcenter_order(embeddings, budget, seed)
        )
        return np.sort(order[:budget])
    if method == "spatial_priority":
        return select_spatial_priority(labels, budget, seed)
    if method == "spatial_balanced":
        return select_spatial_balanced(groups, budget, seed)
    if method == "visit_balanced":
        return select_visit_balanced(
            groups, timestamps, budget, seed, visit_gap_sec
        )
    raise ValueError(f"unknown method {method!r}")


def transform_coordinates(
    lat: np.ndarray,
    lng: np.ndarray,
    *,
    east_m: float,
    north_m: float,
    rotation_deg: float,
    coord_shift_fraction: float,
    coord_permutation_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(lat)
        & np.isfinite(lng)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & (lng >= -180.0)
        & (lng <= 180.0)
    )
    if not bool(np.all(valid)):
        raise ValueError(f"found {int((~valid).sum())} invalid latitude/longitude rows")

    lat0 = float(np.mean(lat))
    lng0 = float(np.mean(lng))
    cos_lat0 = math.cos(math.radians(lat0))
    if abs(cos_lat0) < 1e-8:
        raise ValueError("local tangent transform is unstable at the poles")
    x = np.radians(lng - lng0) * EARTH_RADIUS_M * cos_lat0
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    theta = math.radians(rotation_deg)
    x_out = math.cos(theta) * x - math.sin(theta) * y + east_m
    y_out = math.sin(theta) * x + math.cos(theta) * y + north_m
    lat_out = lat0 + np.degrees(y_out / EARTH_RADIUS_M)
    lng_out = lng0 + np.degrees(x_out / (EARTH_RADIUS_M * cos_lat0))

    if coord_shift_fraction:
        shift = int(round(coord_shift_fraction * lat.size)) % lat.size
        lat_out = np.roll(lat_out, shift)
        lng_out = np.roll(lng_out, shift)
    if coord_permutation_seed is not None:
        permutation = np.random.default_rng(coord_permutation_seed).permutation(lat.size)
        lat_out = lat_out[permutation]
        lng_out = lng_out[permutation]
    return lat_out, lng_out


def h3_cells(lat: np.ndarray, lng: np.ndarray, resolution: int) -> np.ndarray:
    try:
        import h3
    except ImportError as exc:
        raise SystemExit("h3 is required; install requirements-paper.txt") from exc
    return np.asarray(
        [h3.latlng_to_cell(float(a), float(o), resolution) for a, o in zip(lat, lng)],
        dtype=object,
    )


def question_metadata(
    questions: list[dict],
    timestamps: np.ndarray,
    cells: np.ndarray,
    groups: dict[str, np.ndarray],
    visit_gap_sec: float,
) -> dict[str, dict]:
    session_end = float(timestamps[-1]) if timestamps.size else 0.0
    episodes = {
        label: visit_episodes(indices, timestamps, visit_gap_sec)
        for label, indices in groups.items()
    }
    metadata: dict[str, dict] = {}
    exposures: list[float] = []
    for position, question in enumerate(questions):
        qid = str(question.get("id") or f"q{position + 1}")
        intervals = [
            (float(interval[0]), float(interval[1]))
            for interval in question.get("intervals", [])
        ]
        mask = np.zeros(timestamps.size, dtype=bool)
        for start, end in intervals:
            mask |= (timestamps >= start) & (timestamps <= end)
        support = np.flatnonzero(mask)
        if support.size:
            modal_cell = Counter(str(cell) for cell in cells[support]).most_common(1)[0][0]
        elif intervals and timestamps.size:
            midpoint = sum((start + end) / 2.0 for start, end in intervals) / len(intervals)
            nearest = int(np.argmin(np.abs(timestamps - midpoint)))
            modal_cell = str(cells[nearest])
        else:
            modal_cell = None
        exposure = (
            float(groups[modal_cell].size / timestamps.size)
            if modal_cell is not None and timestamps.size
            else None
        )
        if exposure is not None:
            exposures.append(exposure)
        midpoint = (
            sum((start + end) / 2.0 for start, end in intervals) / len(intervals)
            if intervals
            else None
        )
        metadata[qid] = {
            "target_cell": modal_cell,
            "gt_support_frames": int(support.size),
            "place_exposure": exposure,
            "visit_count": len(episodes.get(modal_cell, [])) if modal_cell else 0,
            "normalized_age": (
                (session_end - midpoint) / session_end
                if midpoint is not None and session_end > 0
                else None
            ),
            "old_event": bool(intervals and max(end for _, end in intervals) <= 0.5 * session_end),
            "recent_event": bool(intervals and min(start for start, _ in intervals) >= 0.75 * session_end),
        }
    q25 = float(np.quantile(exposures, 0.25)) if exposures else None
    q75 = float(np.quantile(exposures, 0.75)) if exposures else None
    for item in metadata.values():
        exposure = item["place_exposure"]
        visits = item["visit_count"]
        item["rare_place"] = bool(q25 is not None and exposure is not None and exposure <= q25)
        item["common_place"] = bool(q75 is not None and exposure is not None and exposure >= q75)
        item["revisited_place"] = visits >= 2
        item["heavily_revisited"] = visits >= 4
        item["rare_place_q25"] = q25
        item["common_place_q75"] = q75
    return metadata


def oracle_hit(retained_timestamps: np.ndarray, intervals: list[tuple[float, float]]) -> bool:
    return any(
        bool(np.any((retained_timestamps >= start) & (retained_timestamps <= end)))
        for start, end in intervals
    )


def make_runner(checkpoint: str, device: str):
    package_root = Path(__file__).resolve().parents[1] / "extraction"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from psm_extraction.models import make_runner as make_model_runner

    family = "siglip" if "siglip" in checkpoint.lower() else "clip"
    runner = make_model_runner(
        family, checkpoint=checkpoint, backend="auto", device=device
    )
    print(f"[fixed_budget] {family} runner: {runner.backend}", file=sys.stderr)
    return runner


def software_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for distribution in (
        "numpy",
        "h5py",
        "h3",
        "torch",
        "transformers",
        "psm-extraction",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def resolved_model_revision(runner) -> str | None:
    model = getattr(runner, "_model", None)
    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision) if revision else None


def pip_freeze() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.splitlines() if proc.returncode == 0 else []


def git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def validate_resume_capture(path: Path, expected: dict[str, object]) -> None:
    """Refuse to reuse a capture whose filename-compatible settings differ."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot resume from invalid capture {path}: {exc}") from exc
    transform = data.get("spatial_transform")
    actual: dict[str, object] = {
        "retention_method": data.get("retention_method"),
        "exemplar_budget": data.get("exemplar_budget"),
        "h3_resolution": data.get("h3_resolution"),
        "seed": data.get("psm_seed"),
        "top": data.get("top"),
        "group": data.get("group"),
        "clip_checkpoint": data.get("clip_checkpoint"),
        "clip_backend": data.get("clip_backend"),
        "iou_threshold": data.get("iou_threshold"),
        "exemplar_tolerance": data.get("exemplar_tolerance_sec"),
        "visit_gap_sec": data.get("visit_gap_sec"),
        "hll_capacity": data.get("hll_capacity"),
        "hll_precision": data.get("hll_precision"),
        "features": data.get("features"),
        "questions_file": data.get("questions_file"),
        "features_sha256": data.get("features_sha256"),
        "questions_sha256": data.get("questions_sha256"),
        "git_commit": data.get("git_commit"),
        "evaluation_code_sha256": data.get("evaluation_code_sha256"),
        "software_versions": data.get("software_versions"),
        "model_revision": data.get("model_revision"),
        "feature_checkpoint": data.get("feature_checkpoint"),
        "spatial_transform": transform,
    }
    mismatches = [
        f"{key}: existing={actual.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        details = "; ".join(mismatches)
        raise SystemExit(
            f"refusing stale resume for {path}: {details}. Pass --force to overwrite."
        )


def transform_tag(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.translate_east_m:
        parts.append(f"e{args.translate_east_m:g}")
    if args.translate_north_m:
        parts.append(f"n{args.translate_north_m:g}")
    if args.rotation_deg:
        parts.append(f"rot{args.rotation_deg:g}")
    if args.coord_shift_fraction:
        parts.append(f"null{args.coord_shift_fraction:g}")
    if args.coord_permutation_seed is not None:
        parts.append(f"perm{args.coord_permutation_seed}")
    raw = "_".join(parts) if parts else "base"
    return raw.replace("-", "m").replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--group", default="clip")
    ap.add_argument("--budgets", default="128",
                    help="comma-separated retained-exemplar counts")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--h3-resolutions", default="12")
    ap.add_argument(
        "--methods",
        default=(
            "global_reservoir,fifo,hybrid,uniform_time,semantic_kcenter,"
            "spatial_priority,"
            "spatial_balanced,visit_balanced"
        ),
    )
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--visit-gap-sec", type=float, default=30.0)
    ap.add_argument("--hll-capacity", type=int, default=60)
    ap.add_argument("--hll-precision", type=int, default=10)
    ap.add_argument("--translate-east-m", type=float, default=0.0)
    ap.add_argument("--translate-north-m", type=float, default=0.0)
    ap.add_argument("--rotation-deg", type=float, default=0.0)
    ap.add_argument(
        "--coord-shift-fraction",
        type=float,
        default=0.0,
        help="circularly roll coordinates relative to embeddings; spatial-null control",
    )
    ap.add_argument(
        "--coord-permutation-seed",
        type=int,
        default=None,
        help="randomly permute coordinates relative to embeddings; stronger spatial null",
    )
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    )
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="testing only: allow query checkpoint to differ from HDF5 metadata",
    )
    ap.add_argument("--iou-threshold", type=float, default=None)
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--out-dir", type=Path, default=Path("captures/fixed_budget"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        budgets = parse_int_list(args.budgets, "budgets")
        seeds = parse_int_list(args.seeds, "seeds", minimum=0)
        resolutions = parse_int_list(args.h3_resolutions, "h3-resolutions", minimum=0)
        methods = parse_methods(args.methods)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    if any(resolution > 15 for resolution in resolutions):
        ap.error("H3 resolutions must be in [0, 15]")
    if args.top <= 0 or args.visit_gap_sec <= 0:
        ap.error("--top and --visit-gap-sec must be positive")
    if not 4 <= args.hll_precision <= 18:
        ap.error("--hll-precision must be in [4, 18]")
    if args.hll_capacity <= 0:
        ap.error("--hll-capacity must be positive")
    if not 0.0 <= args.coord_shift_fraction < 1.0:
        ap.error("--coord-shift-fraction must be in [0, 1)")
    if args.coord_shift_fraction and args.coord_permutation_seed is not None:
        ap.error("coordinate shift and permutation controls are mutually exclusive")
    if not args.features.exists() or not args.questions.exists():
        ap.error("features and questions files must exist")

    spec = load_questions(args.questions)
    all_questions = spec.get("questions") or []
    questions = [
        question
        for question in all_questions
        if question.get("query_mode", "similarity_search") == "similarity_search"
        and question.get("query")
        and question.get("intervals")
    ]
    skipped = len(all_questions) - len(questions)
    if not questions:
        raise SystemExit("no scored similarity-search questions")
    question_ids = [
        str(question.get("id") or f"q{position + 1}")
        for position, question in enumerate(questions)
    ]
    duplicate_ids = sorted(
        qid for qid in set(question_ids) if question_ids.count(qid) > 1
    )
    if duplicate_ids:
        raise SystemExit(f"duplicate question IDs: {duplicate_ids}")
    if skipped:
        print(
            f"[fixed_budget] excluding {skipped} non-text or unscored questions",
            file=sys.stderr,
        )
    iou_threshold = (
        float(args.iou_threshold)
        if args.iou_threshold is not None
        else float(spec.get("iou_threshold", 0.3))
    )

    emb_unit, timestamps, session_start = load_features(args.features, args.group)
    n_frames, dim = emb_unit.shape
    semantic_orders = (
        {
            seed: semantic_kcenter_order(emb_unit, max(budgets), seed)
            for seed in seeds
        }
        if "semantic_kcenter" in methods
        else {}
    )
    features_sha256 = sha256_file(args.features)
    questions_sha256 = sha256_file(args.questions)
    with h5py.File(args.features, "r") as handle:
        group = handle[args.group]
        if "lat" not in group or "lng" not in group:
            raise SystemExit(f"{args.features}::{args.group} lacks lat/lng datasets")
        canonical_lat = group["lat"][:].astype(np.float64)
        canonical_lng = group["lng"][:].astype(np.float64)
        feature_checkpoint_raw = group.attrs.get("checkpoint")
    if isinstance(feature_checkpoint_raw, bytes):
        feature_checkpoint = feature_checkpoint_raw.decode("utf-8")
    elif feature_checkpoint_raw is None:
        feature_checkpoint = None
    else:
        feature_checkpoint = str(feature_checkpoint_raw)
    if (
        feature_checkpoint
        and feature_checkpoint != args.clip_checkpoint
        and not args.allow_checkpoint_mismatch
    ):
        raise SystemExit(
            f"query checkpoint {args.clip_checkpoint!r} does not match feature "
            f"checkpoint {feature_checkpoint!r}; pass --allow-checkpoint-mismatch "
            "only for plumbing tests"
        )
    if canonical_lat.shape != (n_frames,) or canonical_lng.shape != (n_frames,):
        raise SystemExit("lat/lng row counts do not match embeddings")
    lat, lng = transform_coordinates(
        canonical_lat,
        canonical_lng,
        east_m=args.translate_east_m,
        north_m=args.translate_north_m,
        rotation_deg=args.rotation_deg,
        coord_shift_fraction=args.coord_shift_fraction,
        coord_permutation_seed=args.coord_permutation_seed,
    )

    tag = transform_tag(args)
    plans: dict[int, tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict]]] = {}
    for resolution in resolutions:
        cells = h3_cells(lat, lng, resolution)
        groups = group_indices(cells)
        canonical_cells = h3_cells(canonical_lat, canonical_lng, resolution)
        canonical_groups = group_indices(canonical_cells)
        metadata = question_metadata(
            questions,
            timestamps,
            canonical_cells,
            canonical_groups,
            args.visit_gap_sec,
        )
        plans[resolution] = (cells, groups, metadata)
        print(
            f"[fixed_budget] r{resolution}: {len(groups)} cells, "
            f"{n_frames} frames, transform={tag}",
            file=sys.stderr,
        )

    if args.dry_run:
        print("method,resolution,budget,seed,retained,cells")
        for resolution, (cells, groups, _) in plans.items():
            for budget in budgets:
                for seed in seeds:
                    for method in methods:
                        selected = select_indices(
                            method,
                            n_frames,
                            budget,
                            seed,
                            cells,
                            groups,
                            timestamps,
                            emb_unit,
                            semantic_orders.get(seed),
                            args.visit_gap_sec,
                        )
                        print(
                            f"{method},{resolution},{budget},{seed},"
                            f"{selected.size},{len(groups)}"
                        )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = make_runner(args.clip_checkpoint, args.clip_device)
    versions = software_versions()
    model_revision = resolved_model_revision(runner)
    query_vectors: dict[str, np.ndarray] = {}
    for position, question in enumerate(questions):
        qid = str(question.get("id") or f"q{position + 1}")
        vector = embed_query_text(runner, str(question["query"]))
        if vector.size != dim:
            raise SystemExit(
                f"query encoder produced {vector.size} dims, but features have {dim}; "
                "use the checkpoint that produced the feature file"
            )
        query_vectors[qid] = vector

    repo_root = Path(__file__).resolve().parents[1]
    commit = git_commit(repo_root)
    code_sha256 = evaluation_code_sha256(repo_root)
    item_bytes = dim * np.dtype(np.float32).itemsize + TIMESTAMP_BYTES
    outputs: list[str] = []
    for resolution, (cells, groups, metadata) in plans.items():
        cell_count = len(groups)
        hll_register_bytes = cell_count * args.hll_capacity * (1 << args.hll_precision)
        for budget in budgets:
            for seed in seeds:
                for method in methods:
                    out_path = args.out_dir / (
                        f"{method}_m{budget}_r{resolution}_s{seed}_{tag}.json"
                    )
                    if out_path.exists() and not args.force:
                        validate_resume_capture(
                            out_path,
                            {
                                "retention_method": method,
                                "exemplar_budget": budget,
                                "h3_resolution": resolution,
                                "seed": seed,
                                "top": args.top,
                                "group": args.group,
                                "clip_checkpoint": args.clip_checkpoint,
                                "clip_backend": getattr(runner, "backend", "unknown"),
                                "iou_threshold": iou_threshold,
                                "exemplar_tolerance": args.exemplar_tolerance,
                                "visit_gap_sec": args.visit_gap_sec,
                                "hll_capacity": args.hll_capacity,
                                "hll_precision": args.hll_precision,
                                "features": str(args.features),
                                "questions_file": str(args.questions),
                                "features_sha256": features_sha256,
                                "questions_sha256": questions_sha256,
                                "git_commit": commit,
                                "evaluation_code_sha256": code_sha256,
                                "software_versions": versions,
                                "model_revision": model_revision,
                                "feature_checkpoint": feature_checkpoint,
                                "spatial_transform": {
                                    "tag": tag,
                                    "translate_east_m": args.translate_east_m,
                                    "translate_north_m": args.translate_north_m,
                                    "rotation_deg": args.rotation_deg,
                                    "coord_shift_fraction": args.coord_shift_fraction,
                                    "coord_permutation_seed": args.coord_permutation_seed,
                                },
                            },
                        )
                        print(f"[fixed_budget] resume: {out_path}", file=sys.stderr)
                        outputs.append(str(out_path))
                        continue
                    selected = select_indices(
                        method,
                        n_frames,
                        budget,
                        seed,
                        cells,
                        groups,
                        timestamps,
                        emb_unit,
                        semantic_orders.get(seed),
                        args.visit_gap_sec,
                    )
                    retained_emb = emb_unit[selected]
                    retained_ts = timestamps[selected]
                    records: list[dict] = []
                    for position, question in enumerate(questions):
                        qid = str(question.get("id") or f"q{position + 1}")
                        intervals = [
                            (float(interval[0]), float(interval[1]))
                            for interval in question["intervals"]
                        ]
                        top_intervals = topk_brute_force(
                            query_vectors[qid],
                            retained_emb,
                            retained_ts,
                            top=args.top,
                            exemplar_tolerance=args.exemplar_tolerance,
                        )
                        preds = score_predictions(
                            top_intervals,
                            intervals,
                            exemplar_tolerance=args.exemplar_tolerance,
                        )
                        record = summarize_question(
                            qid,
                            str(question["query"]),
                            question.get("category") or "(uncategorized)",
                            question.get("notes", ""),
                            intervals,
                            preds,
                            iou_threshold=iou_threshold,
                        )
                        record["oracle_retained_hit"] = oracle_hit(retained_ts, intervals)
                        record["strata"] = metadata[qid]
                        records.append(record)

                    exemplar_bytes = int(selected.size) * item_bytes
                    deployment_hll_bytes = (
                        hll_register_bytes if method in SPATIAL_METHODS else 0
                    )
                    total_logical_bytes = exemplar_bytes
                    write_eval_json(
                        out_path,
                        features_h5=args.features,
                        questions_file=args.questions,
                        group=args.group,
                        top=args.top,
                        records=records,
                        session_start=session_start,
                        clip_checkpoint=args.clip_checkpoint,
                        clip_backend=getattr(runner, "backend", "unknown"),
                        iou_threshold=iou_threshold,
                        exemplar_tolerance=args.exemplar_tolerance,
                        baseline_method=method,
                        seed=seed,
                        extra_settings={
                            "experiment": "fixed_global_exemplar_budget",
                            "retention_method": method,
                            "exemplar_budget": budget,
                            "actual_retained": int(selected.size),
                            "retained_fraction": (
                                float(selected.size / n_frames) if n_frames else 0.0
                            ),
                            "record_count": n_frames,
                            "embedding_dim": dim,
                            "h3_resolution": resolution,
                            "cell_count": cell_count,
                            "visit_gap_sec": args.visit_gap_sec,
                            "hll_capacity": args.hll_capacity,
                            "hll_precision": args.hll_precision,
                            "deployment_hll_register_bytes": deployment_hll_bytes,
                            "selection_protocol": (
                                "offline_final_partition_diagnostic"
                                if method in OFFLINE_SPATIAL_METHODS
                                else (
                                    "causal_streaming"
                                    if method
                                    in {
                                        "global_reservoir",
                                        "fifo",
                                        "hybrid",
                                        "spatial_priority",
                                    }
                                    else "offline_baseline"
                                )
                            ),
                            "git_commit": commit,
                            "evaluation_code_sha256": code_sha256,
                            "software_versions": versions,
                            "model_revision": model_revision,
                            "feature_checkpoint": feature_checkpoint,
                            "features_sha256": features_sha256,
                            "questions_sha256": questions_sha256,
                            "spatial_transform": {
                                "tag": tag,
                                "translate_east_m": args.translate_east_m,
                                "translate_north_m": args.translate_north_m,
                                "rotation_deg": args.rotation_deg,
                                "coord_shift_fraction": args.coord_shift_fraction,
                                "coord_permutation_seed": args.coord_permutation_seed,
                            },
                            "fixed_budget": {
                                "kind": "global_exemplar_count",
                                "target_exemplars": budget,
                                "actual_exemplars": int(selected.size),
                                "item_bytes": item_bytes,
                                "exemplar_logical_bytes": exemplar_bytes,
                                "hll_register_bytes": 0,
                                "total_logical_state_bytes": total_logical_bytes,
                                "accounting": (
                                    "retained raw float32 payload + float64 timestamp; "
                                    "container, per-cell counters, hash-table, allocator, "
                                    "and optional deployment HLL overhead excluded"
                                ),
                            },
                        },
                    )
                    outputs.append(str(out_path))
                    if args.verbose:
                        print(
                            f"[fixed_budget] {method} r{resolution} M={budget} "
                            f"seed={seed} -> {out_path.name}",
                            file=sys.stderr,
                        )

    close = getattr(runner, "close", None)
    if close is not None:
        close()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "evaluation_code_sha256": code_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": versions,
        "model_revision": model_revision,
        "feature_checkpoint": feature_checkpoint,
        "pip_freeze": pip_freeze(),
        "features": str(args.features.resolve()),
        "questions": str(args.questions.resolve()),
        "features_sha256": features_sha256,
        "questions_sha256": questions_sha256,
        "group": args.group,
        "methods": methods,
        "budgets": budgets,
        "seeds": seeds,
        "h3_resolutions": resolutions,
        "top": args.top,
        "spatial_transform": tag,
        "outputs": outputs,
    }
    (args.out_dir / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fixed_budget] wrote {len(outputs)} captures under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
