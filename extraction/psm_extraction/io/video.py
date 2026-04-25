"""ffmpeg-backed frame reader.

Spawns one `ffmpeg` subprocess per video, samples at the requested fps, and
writes JPEGs into a target directory. Identical wire format to the existing
`scripts/e5_clip_demo.py` so caches built by the demo remain compatible.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _check_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"required executable {name!r} not found on PATH")


def _run(cmd: list[str], *, verbose: bool) -> subprocess.CompletedProcess:
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def video_duration(video_path: Path, *, verbose: bool = False) -> float | None:
    """Return the video duration in seconds, or None if ffprobe is unavailable."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = _run(cmd, verbose=verbose)
    except RuntimeError:
        return None
    payload = json.loads(result.stdout)
    raw = payload.get("format", {}).get("duration")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def extract_frames(
    video_path: Path,
    sample_fps: float,
    output_dir: Path,
    *,
    verbose: bool = False,
) -> list[Path]:
    """Sample frames at `sample_fps` into `output_dir/frame_%06d.jpg`.

    The output directory is wiped and recreated to keep the indexing simple.
    Returns the sorted list of produced JPEG paths.
    """
    _check_tool("ffmpeg")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps}",
        "-start_number",
        "0",
        str(output_dir / "frame_%06d.jpg"),
    ]
    _run(cmd, verbose=verbose)
    paths = sorted(output_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError(f"ffmpeg produced no frames from {video_path}")
    return paths
