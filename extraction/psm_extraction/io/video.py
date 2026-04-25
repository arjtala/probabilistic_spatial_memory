"""ffmpeg-backed frame reader.

Spawns one `ffmpeg` subprocess per video, samples at the requested fps, and
writes JPEGs into a target directory. Identical wire format to the existing
`scripts/e5_clip_demo.py` so caches built by the demo remain compatible.

A small `.extract_manifest.json` is written next to the JPEGs recording the
source video path and `sample_fps`. Subsequent calls with matching params
reuse the cache instead of re-running ffmpeg; pass `force=True` to wipe and
re-extract anyway (or delete the manifest by hand).
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


_MANIFEST_FILENAME = ".extract_manifest.json"


def _read_manifest(output_dir: Path) -> dict | None:
    p = output_dir / _MANIFEST_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(
    output_dir: Path, *, video_path: Path, sample_fps: float, frame_count: int
) -> None:
    (output_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "video": str(video_path.resolve()),
                "sample_fps": float(sample_fps),
                "frame_count": int(frame_count),
            },
            indent=2,
        )
    )


def extract_frames(
    video_path: Path,
    sample_fps: float,
    output_dir: Path,
    *,
    verbose: bool = False,
    force: bool = False,
) -> list[Path]:
    """Sample frames at `sample_fps` into `output_dir/frame_%06d.jpg`.

    Reuses an existing cache when the recorded manifest matches the current
    `video_path` and `sample_fps`; pass `force=True` to wipe the cache and
    re-run ffmpeg even if the manifest matches. Returns the sorted list of
    JPEG paths either way.
    """
    _check_tool("ffmpeg")

    if not force:
        existing_manifest = _read_manifest(output_dir)
        if (
            existing_manifest is not None
            and existing_manifest.get("video") == str(video_path.resolve())
            and float(existing_manifest.get("sample_fps", 0.0)) == float(sample_fps)
        ):
            cached = sorted(output_dir.glob("frame_*.jpg"))
            if cached and len(cached) == int(existing_manifest.get("frame_count", -1)):
                if verbose:
                    print(
                        f"[frames] reusing {len(cached)} cached JPEGs at "
                        f"{output_dir} (video={video_path.name}, fps={sample_fps})",
                        file=sys.stderr,
                    )
                return cached

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
    _write_manifest(
        output_dir,
        video_path=video_path,
        sample_fps=sample_fps,
        frame_count=len(paths),
    )
    return paths
