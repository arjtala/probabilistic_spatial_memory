"""Find Ego-Exo4D takes with enough questions + duration for a useful smoke test.

Walks the questions output root, filters to takes with [100..300] questions,
runs ffprobe on each ego MP4 to get duration. Stops at the first take longer
than `MIN_DUR_S`. Prints `<take>: <n_q> questions, <duration>` per match.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/egoexo4d_atomic")
TAKES = Path("/datasets/egoexo4d/v2/takes")
MIN_DUR_S = 600.0   # 10 min
MIN_Q, MAX_Q = 100, 300


def main() -> None:
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue  # skip .take_uids.txt and other non-take files
        qfile = d / "questions.yaml"
        if not qfile.is_file():
            continue
        n = sum(
            1 for line in qfile.read_text().splitlines()
            if line.lstrip().startswith("- id:")
        )
        if not (MIN_Q <= n <= MAX_Q):
            continue
        fav = TAKES / d.name / "frame_aligned_videos"
        mp4 = next(fav.glob("aria*_214-1.mp4"), None) if fav.exists() else None
        if mp4 is None:
            continue
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(mp4)],
            capture_output=True, text=True,
        )
        try:
            dur = float(out.stdout.strip())
        except ValueError:
            continue
        print(f"{d.name}: {n} questions, {dur:.0f}s ({dur/60:.1f}min)")
        if dur >= MIN_DUR_S:
            break


if __name__ == "__main__":
    main()
