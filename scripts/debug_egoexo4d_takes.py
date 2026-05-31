"""Debug: why does the converter still emit UUID dirs?

Run from the repo root:
  python scripts/debug_egoexo4d_takes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extraction"))

from psm_extraction.io import load_take_uid_to_name, read_atomic_descriptions

TAKES_JSON = Path("/datasets/egoexo4d/v2/takes.json")
ATOMIC = Path("/datasets/egoexo4d/v2/annotations/atomic_descriptions_val.json")
SAMPLE_UID = "001ae9a5-9c8a-4710-9f7f-7dc67597a02f"


def main() -> int:
    print(f"takes.json exists: {TAKES_JSON.exists()}")
    if not TAKES_JSON.exists():
        print("FATAL: takes.json missing")
        return 1
    m = load_take_uid_to_name(TAKES_JSON)
    print(f"mapping size: {len(m)}")
    print(f"sample lookup {SAMPLE_UID!r} -> {m.get(SAMPLE_UID)!r}")

    # Show a few entries from the mapping
    items = list(m.items())[:3]
    print("first 3 mapping entries:")
    for uid, name in items:
        print(f"  {uid} -> {name}")

    print()
    print("read_atomic_descriptions with takes_json_path:")
    takes = read_atomic_descriptions(ATOMIC, takes_json_path=TAKES_JSON)
    print(f"n takes: {len(takes)}")
    for t in takes[:3]:
        print(f"  uid={t.take_uid} name={t.take_name!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
