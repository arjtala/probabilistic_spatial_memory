#!/usr/bin/env python3
"""Summarize fixed-budget retrieval evaluation JSON files as Markdown.

The preferred input schema is produced by ``eval_fixed_budget.py`` and adds
the following top-level fields to the usual ``eval_lookback.py`` output:

``retention_method``, ``exemplar_budget``, ``actual_retained``,
``h3_resolution``, ``fixed_budget``, and ``spatial_transform``.

For compatibility with the standalone global-reservoir baseline, this script
also accepts ``baseline_method``, ``reservoir_capacity``,
``retained_exemplars``, and ``psm_seed`` aliases. Missing optional strata or
budget accounting are rendered as ``n/a``; inconsistent accounting is never
silently included.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MIB = 1024 * 1024
STRATA = (
    "rare_place",
    "common_place",
    "old_event",
    "recent_event",
    "revisited_place",
)


class SchemaError(ValueError):
    """An input file is not a valid fixed-budget evaluation record."""


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SchemaError(f"{label} must be an integer, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{label} must be an integer, got {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise SchemaError(f"{label} must be an integer, got {value!r}")
    result = int(number)
    if result < minimum:
        raise SchemaError(f"{label} must be >= {minimum}, got {result}")
    return result


def _optional_integer(value: Any, label: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=minimum)


def _metric(value: Any, label: str) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise SchemaError(f"{label} must be finite, got {value!r}")
    return result


def _flag(value: Any) -> bool | None:
    """Parse a JSON boolean-like value without treating arbitrary text as true."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _canonical_transform(value: Any) -> str:
    if value is None or value == "":
        return "none"
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("tag"):
        return str(value["tag"])
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, order=True)
class GroupKey:
    retention_method: str
    exemplar_budget: int | None
    h3_resolution: int | None
    seed: int | None
    spatial_transform: str
    top: int


@dataclass
class BudgetInfo:
    target_bytes: int | None
    max_logical_bytes: int | None
    logical_bytes: int | None
    item_bytes: int | None


@dataclass
class Run:
    path: Path
    session: str
    key: GroupKey
    top: int
    actual_retained: int | None
    retained_fraction: float | None
    cell_count: int | None
    deployment_hll_bytes: int | None
    budget: BudgetInfo
    records: list[dict[str, Any]]


@dataclass
class Aggregate:
    tops: set[int] = field(default_factory=set)
    files: int = 0
    n_scored: int = 0
    hits: list[float] = field(default_factory=list)
    mious: list[float] = field(default_factory=list)
    oracle_hits: list[float] = field(default_factory=list)
    stratum_hits: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in STRATA}
    )
    actual_retained: list[int] = field(default_factory=list)
    retained_fractions: list[float] = field(default_factory=list)
    cell_counts: list[int] = field(default_factory=list)
    cells_per_budget: list[float] = field(default_factory=list)
    logical_bytes: list[int] = field(default_factory=list)
    deployment_hll_bytes: list[int] = field(default_factory=list)

    def add(self, run: Run) -> None:
        self.tops.add(run.top)
        self.files += 1
        if run.actual_retained is not None:
            self.actual_retained.append(run.actual_retained)
        if run.retained_fraction is not None:
            self.retained_fractions.append(run.retained_fraction)
        if run.cell_count is not None:
            self.cell_counts.append(run.cell_count)
            if run.key.exemplar_budget:
                self.cells_per_budget.append(
                    run.cell_count / run.key.exemplar_budget
                )
        if run.budget.logical_bytes is not None:
            self.logical_bytes.append(run.budget.logical_bytes)
        if run.deployment_hll_bytes is not None:
            self.deployment_hll_bytes.append(run.deployment_hll_bytes)

        run_hits: list[float] = []
        run_mious: list[float] = []
        run_oracle: list[float] = []
        run_strata: dict[str, list[float]] = {name: [] for name in STRATA}
        for index, record in enumerate(run.records):
            # Match eval_lookback's convention: records without GT intervals
            # are negative controls and do not enter retrieval means.
            if not record.get("intervals_gt"):
                continue
            self.n_scored += 1
            hit_value = _flag(record.get("exemplar_hit_at_k"))
            if hit_value is not None:
                hit = float(hit_value)
                run_hits.append(hit)
            else:
                hit = None

            if record.get("exemplar_iou_at_k") is not None:
                run_mious.append(
                    _metric(
                        record["exemplar_iou_at_k"],
                        f"records[{index}].exemplar_iou_at_k",
                    )
                )

            oracle = _flag(record.get("oracle_retained_hit"))
            if oracle is not None:
                run_oracle.append(float(oracle))

            if hit is None:
                continue
            for name in STRATA:
                if _record_in_stratum(record, name):
                    run_strata[name].append(hit)

        # Macro-average sessions/files. Pooling thousands of questions would
        # let long sessions dominate the headline comparison.
        if run_hits:
            self.hits.append(statistics.fmean(run_hits))
        if run_mious:
            self.mious.append(statistics.fmean(run_mious))
        if run_oracle:
            self.oracle_hits.append(statistics.fmean(run_oracle))
        for name, values in run_strata.items():
            if values:
                self.stratum_hits[name].append(statistics.fmean(values))


def _record_in_stratum(record: dict[str, Any], name: str) -> bool:
    strata = record.get("strata")
    if isinstance(strata, dict):
        parsed = _flag(strata.get(name))
        if parsed is not None:
            return parsed

    # Compatibility with likely stand-alone analysis fields.
    parsed = _flag(record.get(name))
    if parsed is not None:
        return parsed
    if name == "rare_place":
        rarity = str(record.get("rarity_stratum") or "").strip().lower()
        if rarity:
            return rarity in {"rare", "rare_place", "rare-place"}

    # Some early captures encoded explicit stratum labels as categories.
    category = str(record.get("category") or "").strip().lower()
    normalized = category.replace("-", "_").replace(" ", "_")
    return normalized == name or name in normalized.split("+")


def _budget_info(
    data: dict[str, Any],
    *,
    actual_retained: int | None,
    exemplar_budget: int | None,
) -> BudgetInfo:
    raw = data.get("fixed_budget")
    if raw is None:
        raw_map: dict[str, Any] = {}
    elif isinstance(raw, dict):
        raw_map = raw
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw_map = {"target_bytes": raw}
    else:
        raise SchemaError("fixed_budget must be an object or byte count")

    # Preferred eval_fixed_budget.py schema: the hard constraint is exemplar
    # count. HLL register bytes are reported, but deliberately do not reduce
    # another method's exemplar allowance.
    count_schema = raw_map.get("kind") == "global_exemplar_count" or any(
        name in raw_map
        for name in (
            "target_exemplars",
            "actual_exemplars",
            "exemplar_logical_bytes",
            "hll_register_bytes",
            "total_logical_state_bytes",
        )
    )
    if count_schema:
        required = (
            "target_exemplars",
            "actual_exemplars",
            "item_bytes",
            "exemplar_logical_bytes",
            "hll_register_bytes",
            "total_logical_state_bytes",
        )
        missing = [name for name in required if raw_map.get(name) is None]
        if missing:
            raise SchemaError(
                "fixed_budget global-exemplar-count schema is missing: "
                + ", ".join(missing)
            )
        kind = raw_map.get("kind")
        if kind is not None and kind != "global_exemplar_count":
            raise SchemaError(f"unsupported fixed_budget.kind {kind!r}")

        target_exemplars = _integer(
            raw_map["target_exemplars"], "fixed_budget.target_exemplars"
        )
        actual_exemplars = _integer(
            raw_map["actual_exemplars"], "fixed_budget.actual_exemplars"
        )
        item = _integer(raw_map["item_bytes"], "fixed_budget.item_bytes", minimum=1)
        exemplar_bytes = _integer(
            raw_map["exemplar_logical_bytes"],
            "fixed_budget.exemplar_logical_bytes",
        )
        hll_bytes = _integer(
            raw_map["hll_register_bytes"], "fixed_budget.hll_register_bytes"
        )
        total_bytes = _integer(
            raw_map["total_logical_state_bytes"],
            "fixed_budget.total_logical_state_bytes",
        )

        if actual_exemplars > target_exemplars:
            raise SchemaError(
                "fixed_budget.actual_exemplars exceeds target_exemplars "
                f"({actual_exemplars} > {target_exemplars})"
            )
        if exemplar_budget is not None and target_exemplars != exemplar_budget:
            raise SchemaError(
                "fixed_budget.target_exemplars disagrees with exemplar_budget "
                f"({target_exemplars} != {exemplar_budget})"
            )
        if actual_retained is not None and actual_exemplars != actual_retained:
            raise SchemaError(
                "fixed_budget.actual_exemplars disagrees with actual_retained "
                f"({actual_exemplars} != {actual_retained})"
            )
        expected_exemplar_bytes = actual_exemplars * item
        if exemplar_bytes != expected_exemplar_bytes:
            raise SchemaError(
                "fixed_budget.exemplar_logical_bytes is inconsistent: "
                f"{exemplar_bytes} != {actual_exemplars} * {item}"
            )
        if total_bytes != exemplar_bytes + hll_bytes:
            raise SchemaError(
                "fixed_budget.total_logical_state_bytes is inconsistent: "
                f"{total_bytes} != {exemplar_bytes} + {hll_bytes}"
            )
        return BudgetInfo(None, None, total_bytes, item)

    target = _optional_integer(
        _first(raw_map, ("target_bytes", "budget_bytes", "logical_budget_bytes")),
        "fixed_budget.target_bytes",
    )
    maximum = _optional_integer(
        _first(
            raw_map,
            ("max_logical_state_bytes", "max_logical_bytes", "maximum_logical_bytes"),
        ),
        "fixed_budget.max_logical_state_bytes",
    )
    logical = _optional_integer(
        _first(
            raw_map,
            (
                "total_logical_state_bytes",
                "logical_state_bytes",
                "logical_bytes",
                "actual_logical_bytes",
            ),
        ),
        "fixed_budget.logical_state_bytes",
    )
    item = _optional_integer(
        _first(raw_map, ("item_bytes", "bytes_per_exemplar")),
        "fixed_budget.item_bytes",
        minimum=1,
    )

    if maximum is not None and target is not None and maximum > target:
        raise SchemaError(
            "fixed_budget.max_logical_state_bytes exceeds target_bytes "
            f"({maximum} > {target})"
        )
    ceiling = maximum if maximum is not None else target
    if logical is not None and ceiling is not None and logical > ceiling:
        raise SchemaError(
            "fixed_budget.logical_state_bytes exceeds its budget ceiling "
            f"({logical} > {ceiling})"
        )
    if actual_retained is not None and exemplar_budget is not None:
        if actual_retained > exemplar_budget:
            raise SchemaError(
                f"actual_retained exceeds exemplar_budget "
                f"({actual_retained} > {exemplar_budget})"
            )
    if item is not None and actual_retained is not None and logical is not None:
        item_state = item * actual_retained
        # A producer may charge additional cell/index state, but it cannot
        # report fewer logical bytes than its retained items require.
        if logical < item_state:
            raise SchemaError(
                "fixed_budget.logical_state_bytes undercounts retained items "
                f"({logical} < {actual_retained} * {item})"
            )
        accounting = str(raw_map.get("accounting") or "").lower()
        item_only = (
            "container overhead excluded" in accounting
            and not any(token in accounting for token in ("hll", "cell", "index"))
        )
        if item_only and logical != item_state:
            raise SchemaError(
                "item-only fixed-budget accounting is inconsistent: "
                f"logical_state_bytes={logical}, expected {item_state}"
            )
    if item is not None and exemplar_budget is not None and maximum is not None:
        minimum_maximum = item * exemplar_budget
        if maximum < minimum_maximum:
            raise SchemaError(
                "fixed_budget.max_logical_state_bytes cannot hold exemplar_budget "
                f"items ({maximum} < {exemplar_budget} * {item})"
            )

    return BudgetInfo(target, maximum, logical, item)


def _load_run(path: Path, *, strict: bool) -> Run:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise SchemaError(f"cannot read file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise SchemaError("top-level JSON value must be an object")

    method = _first(data, ("retention_method", "baseline_method"))
    if method is None:
        if strict:
            raise SchemaError("missing retention_method")
        method = "psm"
    method = str(method).strip()
    if not method:
        raise SchemaError("retention_method must not be empty")

    raw_budget = _first(data, ("exemplar_budget", "reservoir_capacity"))
    fixed = data.get("fixed_budget")
    if raw_budget is None and isinstance(fixed, dict):
        raw_budget = fixed.get("target_exemplars")
        if raw_budget is None:
            maximum = _first(
                fixed, ("max_logical_state_bytes", "max_logical_bytes")
            )
            item = _first(fixed, ("item_bytes", "bytes_per_exemplar"))
            if maximum is not None and item is not None:
                item_int = _integer(item, "fixed_budget.item_bytes", minimum=1)
                raw_budget = _integer(
                    maximum, "fixed_budget.max_logical_state_bytes"
                ) // item_int
    exemplar_budget = _optional_integer(raw_budget, "exemplar_budget")

    raw_actual = _first(
        data,
        ("actual_retained", "retained_exemplars", "retained_count"),
    )
    if raw_actual is None and isinstance(fixed, dict):
        raw_actual = fixed.get("actual_exemplars")
    actual_retained = _optional_integer(
        raw_actual,
        "actual_retained",
    )
    retained_fraction_raw = data.get("retained_fraction")
    retained_fraction = (
        _metric(retained_fraction_raw, "retained_fraction")
        if retained_fraction_raw is not None
        else None
    )
    if retained_fraction is not None and not 0.0 <= retained_fraction <= 1.0:
        raise SchemaError("retained_fraction must be in [0, 1]")
    cell_count = _optional_integer(data.get("cell_count"), "cell_count")
    deployment_hll_bytes = _optional_integer(
        data.get("deployment_hll_register_bytes"),
        "deployment_hll_register_bytes",
    )
    h3_resolution = _optional_integer(
        data.get("h3_resolution"), "h3_resolution"
    )
    seed = _optional_integer(_first(data, ("seed", "psm_seed")), "seed")
    transform = _canonical_transform(data.get("spatial_transform"))
    top = _integer(data.get("top", 5), "top", minimum=1)

    records = data.get("records")
    if records is None:
        records = data.get("questions_out")
    if not isinstance(records, list):
        raise SchemaError("records must be a list")
    if not all(isinstance(record, dict) for record in records):
        raise SchemaError("every records entry must be an object")

    budget = _budget_info(
        data,
        actual_retained=actual_retained,
        exemplar_budget=exemplar_budget,
    )
    if strict:
        missing = []
        if exemplar_budget is None:
            missing.append("exemplar_budget")
        if actual_retained is None:
            missing.append("actual_retained")
        if budget.logical_bytes is None:
            missing.append("fixed_budget.logical_state_bytes")
        if missing:
            raise SchemaError("missing required field(s): " + ", ".join(missing))

    return Run(
        path=path,
        session=(
            Path(str(data["features"])).parent.name
            if data.get("features")
            else path.parent.name
        ),
        key=GroupKey(
            retention_method=method,
            exemplar_budget=exemplar_budget,
            h3_resolution=h3_resolution,
            seed=seed,
            spatial_transform=transform,
            top=top,
        ),
        top=top,
        actual_retained=actual_retained,
        retained_fraction=retained_fraction,
        cell_count=cell_count,
        deployment_hll_bytes=deployment_hll_bytes,
        budget=budget,
        records=records,
    )


def _expand_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*.json") if recursive else path.glob("*.json")
            expanded.extend(
                sorted(
                    p
                    for p in iterator
                    if p.is_file() and not p.name.startswith("manifest_")
                )
            )
        else:
            expanded.append(path)
    # Preserve input order while avoiding accidental double aggregation.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        normalized = path.resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def _mean(values: list[float] | list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def _pct(values: list[float], *, show_n: bool = False) -> str:
    value = _mean(values)
    if value is None:
        return "n/a"
    text = f"{value:.1%}"
    return f"{text} (n={len(values)})" if show_n else text


def _decimal(values: list[float]) -> str:
    value = _mean(values)
    return "n/a" if value is None else f"{value:.3f}"


def _slots(values: list[int]) -> str:
    value = _mean(values)
    if value is None:
        return "n/a"
    if all(item == values[0] for item in values):
        return str(values[0])
    return f"{value:.1f}"


def _logical_mib(values: list[int]) -> str:
    value = _mean(values)
    return "n/a" if value is None else f"{value / MIB:.3f}"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(groups: dict[GroupKey, Aggregate], n_inputs: int) -> str:
    all_tops = sorted({key.top for key in groups})
    mixed_top = len(all_tops) > 1
    hit_label = "Hit@K" if mixed_top else f"Hit@{all_tops[0]}"
    miou_label = "mIoU@K" if mixed_top else f"mIoU@{all_tops[0]}"

    lines = [
        "# Fixed-Budget Retrieval Summary",
        "",
        f"_{n_inputs} JSON file(s), {len(groups)} configuration group(s). "
        "Metrics are macro-averaged over session files within each "
        "method/budget/resolution/seed/transform group._",
        "",
    ]
    columns = ["method", "budget", "H3", "seed", "transform"]
    if mixed_top:
        columns.append("K")
    columns.extend(
        [
            "n",
            hit_label,
            miou_label,
            "oracle retained",
            "rare-place Hit",
            "common-place Hit",
            "old-event Hit",
            "recent-event Hit",
            "revisit Hit",
            "actual slots",
            "M/N",
            "cells",
            "C/M",
            "logical MiB",
            "deploy HLL MiB",
        ]
    )
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))

    def sort_key(item: tuple[GroupKey, Aggregate]) -> tuple[Any, ...]:
        key, _ = item
        return (
            key.retention_method,
            -1 if key.exemplar_budget is None else key.exemplar_budget,
            -1 if key.h3_resolution is None else key.h3_resolution,
            -1 if key.seed is None else key.seed,
            key.spatial_transform,
            key.top,
        )

    for key, aggregate in sorted(groups.items(), key=sort_key):
        if len(aggregate.tops) != 1:
            tops = ",".join(str(k) for k in sorted(aggregate.tops))
        else:
            tops = str(next(iter(aggregate.tops)))
        row = [
            f"`{_cell(key.retention_method)}`",
            _cell(key.exemplar_budget),
            _cell(key.h3_resolution),
            _cell(key.seed),
            f"`{_cell(key.spatial_transform)}`",
        ]
        if mixed_top:
            row.append(tops)
        row.extend(
            [
                str(aggregate.n_scored),
                _pct(aggregate.hits),
                _decimal(aggregate.mious),
                _pct(aggregate.oracle_hits),
                _pct(aggregate.stratum_hits["rare_place"], show_n=True),
                _pct(aggregate.stratum_hits["common_place"], show_n=True),
                _pct(aggregate.stratum_hits["old_event"], show_n=True),
                _pct(aggregate.stratum_hits["recent_event"], show_n=True),
                _pct(aggregate.stratum_hits["revisited_place"], show_n=True),
                _slots(aggregate.actual_retained),
                _pct(aggregate.retained_fractions),
                _slots(aggregate.cell_counts),
                _decimal(aggregate.cells_per_budget),
                _logical_mib(aggregate.logical_bytes),
                _logical_mib(aggregate.deployment_hll_bytes),
            ]
        )
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "_Logical MiB is retained selector state; deploy HLL MiB is the separately "
            "modeled PSM ring-register cost. Both may exclude container or allocator "
            "overhead. `n/a` means the source JSON did not provide "
            "that optional metric or stratum; stratum `n` is the number of "
            "session files contributing to that macro-average._",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="evaluation JSON files or directories containing JSON files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="also write the Markdown summary to this path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="search input directories recursively for JSON files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on the first invalid or incomplete fixed-budget input",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print Markdown to stdout (requires --out)",
    )
    args = parser.parse_args()

    if args.quiet and args.out is None:
        parser.error("--quiet requires --out")

    paths = _expand_inputs(args.inputs, args.recursive)
    if not paths:
        print("error: no JSON inputs found", file=sys.stderr)
        return 2

    groups: dict[GroupKey, Aggregate] = defaultdict(Aggregate)
    seen_runs: set[tuple[GroupKey, str]] = set()
    accepted = 0
    skipped = 0
    for path in paths:
        if not path.is_file():
            message = f"{path}: file not found"
            if args.strict:
                print(f"error: {message}", file=sys.stderr)
                return 2
            print(f"warning: {message}; skipping", file=sys.stderr)
            skipped += 1
            continue
        try:
            run = _load_run(path, strict=args.strict)
            identity = (run.key, run.session)
            if identity in seen_runs:
                raise SchemaError(
                    f"duplicate session/config capture for {run.session}: {run.key}"
                )
            seen_runs.add(identity)
            groups[run.key].add(run)
        except (SchemaError, KeyError) as exc:
            if args.strict:
                print(f"error: {path}: {exc}", file=sys.stderr)
                return 2
            print(f"warning: {path}: {exc}; skipping", file=sys.stderr)
            skipped += 1
            continue
        accepted += 1

    if not groups:
        print("error: no valid fixed-budget evaluation inputs", file=sys.stderr)
        return 2

    summary = _markdown(groups, accepted)
    if not args.quiet:
        print(summary, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary)
        print(f"[fixed-budget] wrote {args.out}", file=sys.stderr)
    if skipped:
        print(
            f"[fixed-budget] accepted {accepted} file(s), skipped {skipped}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
