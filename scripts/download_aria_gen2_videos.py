#!/usr/bin/env python3
"""Download Aria Gen 2 Pilot Dataset video_main_rgb MP4s for local annotation.

The full VRS bundles are 2-3 GB each; the `video_main_rgb` preview MP4s
are 470 MB - 1.25 GB each, total ~10 GB across the 12 sessions. Enough
to scrub through in a browser without pulling the full VRS.

Reads `datasets/AriaGen2PilotDataset_download_urls.json` and pulls only
the `video_main_rgb` entry per sequence. Skips files that are already
on disk + match the expected SHA-1. Re-runnable: idempotent on a
partial download.

Usage:
    python scripts/download_aria_gen2_videos.py
    python scripts/download_aria_gen2_videos.py --out annotation/aria_gen2_videos
    python scripts/download_aria_gen2_videos.py --only walk_0,walk_1   # subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


_DEFAULT_URLS_JSON = Path("datasets/AriaGen2PilotDataset_download_urls.json")
_DEFAULT_OUT = Path("annotation/aria_gen2_videos")


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dst: Path, expected_bytes: int) -> None:
    """Stream-download `url` to `dst`, printing a progress bar.

    Writes to a `.part` file and renames on success so a Ctrl-C
    mid-download doesn't leave a half-file that future runs treat as
    "already downloaded."
    """
    tmp = dst.with_suffix(dst.suffix + ".part")
    written = 0
    last_pct = -1
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if expected_bytes > 0:
                pct = int(100 * written / expected_bytes)
                if pct != last_pct:
                    bar_len = 40
                    filled = int(bar_len * written / expected_bytes)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    sys.stderr.write(
                        f"\r  [{bar}] {pct:3d}%  "
                        f"{written/1e6:7.1f}/{expected_bytes/1e6:.1f} MB"
                    )
                    sys.stderr.flush()
                    last_pct = pct
    sys.stderr.write("\n")
    tmp.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--urls-json", type=Path, default=_DEFAULT_URLS_JSON,
        help="Aria Gen 2 download-URLs JSON (default: datasets/...)",
    )
    ap.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output directory; one MP4 per sequence (default: annotation/...)",
    )
    ap.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated sequence names to include; default = all 12.",
    )
    ap.add_argument(
        "--skip-sha", action="store_true",
        help="Trust file size only; don't recompute SHA-1 on already-downloaded files.",
    )
    args = ap.parse_args()

    if not args.urls_json.exists():
        print(f"ERR: {args.urls_json} not found", file=sys.stderr)
        return 1

    payload = json.loads(args.urls_json.read_text())
    sequences = payload.get("sequences") or {}
    if not sequences:
        print("ERR: no 'sequences' in URLs JSON", file=sys.stderr)
        return 1

    wanted = set(args.only.split(",")) if args.only else None
    targets = sorted(sequences.keys())
    if wanted is not None:
        targets = [n for n in targets if n in wanted]
        missing = wanted - set(targets)
        if missing:
            print(f"WARN: requested but not in JSON: {sorted(missing)}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)

    total_to_fetch = 0
    plan: list[tuple[str, str, Path, int, str]] = []
    for name in targets:
        seq = sequences[name]
        v = seq.get("video_main_rgb")
        if not v:
            print(f"  {name}: no video_main_rgb entry, skipping", file=sys.stderr)
            continue
        fn = v.get("filename")
        url = v.get("download_url")
        size = int(v.get("file_size_bytes") or 0)
        sha = v.get("sha1sum") or ""
        if not fn or not url:
            print(f"  {name}: missing filename or download_url, skipping",
                  file=sys.stderr)
            continue
        dst = args.out / fn
        # Skip if file is on disk + size matches (+ optional SHA check).
        if dst.exists() and dst.stat().st_size == size:
            if args.skip_sha:
                print(f"  {name}: already present ({size/1e6:.1f} MB), trusting size",
                      file=sys.stderr)
                continue
            actual_sha = _sha1(dst)
            if actual_sha == sha:
                print(f"  {name}: already present + SHA matches ({size/1e6:.1f} MB)",
                      file=sys.stderr)
                continue
            else:
                print(f"  {name}: size matches but SHA differs; re-downloading",
                      file=sys.stderr)
        plan.append((name, url, dst, size, sha))
        total_to_fetch += size

    if not plan:
        print(f"[aria-dl] nothing to fetch; {len(targets)} target(s) already on disk",
              file=sys.stderr)
        return 0

    print(f"[aria-dl] fetching {len(plan)} sequence(s), "
          f"~{total_to_fetch/1e9:.2f} GB total -> {args.out}",
          file=sys.stderr)

    for i, (name, url, dst, size, sha) in enumerate(plan, start=1):
        print(f"[{i}/{len(plan)}] {name} ({size/1e6:.1f} MB) -> {dst.name}",
              file=sys.stderr)
        try:
            _download_with_progress(url, dst, size)
        except KeyboardInterrupt:
            print("\n[aria-dl] interrupted; re-run to resume (partial file kept as .part)",
                  file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001 — surface any urllib/IO error.
            print(f"[aria-dl] {name}: download failed: {exc}", file=sys.stderr)
            return 1
        if sha and not args.skip_sha:
            actual = _sha1(dst)
            if actual != sha:
                print(f"[aria-dl] {name}: SHA mismatch "
                      f"(expected {sha}, got {actual}); kept as {dst.name}",
                      file=sys.stderr)
                return 1

    print(f"[aria-dl] done. {len(plan)} file(s) under {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
