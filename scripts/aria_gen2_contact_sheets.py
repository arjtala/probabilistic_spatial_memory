#!/usr/bin/env python3
"""Generate annotation-friendly HTML contact sheets for Aria Gen 2 videos.

For each MP4 under `--videos`, extracts one keyframe every
`--stride-sec` seconds (default 5s) and writes a self-contained HTML
under `--out` that shows:

  - A pinned <video> element at the top, playing the local MP4.
  - A scrollable grid of thumbnails, each labeled with its timestamp.
  - Clicking a thumbnail seeks the <video> to that timestamp.
  - A timestamp readout below the video, hover-copyable, formatted as
    a YAML interval scaffold ("- [t-1.5, t+1.5]") so authoring a
    questions.yaml entry is paste-paste-type.

The HTML is one file per session, embedding nothing — references the
MP4 and JPEG paths *relative* to the HTML's own location so the whole
output directory is movable.

Usage:
    python scripts/aria_gen2_contact_sheets.py
    python scripts/aria_gen2_contact_sheets.py \\
        --videos annotation/aria_gen2_videos \\
        --out annotation/contact_sheets \\
        --stride-sec 5 --thumb-width 240

Output layout:
    annotation/contact_sheets/
      walk_0.html
      walk_0_thumbs/
        t000.000.jpg
        t005.000.jpg
        ...
      walk_1.html
      walk_1_thumbs/...

Open each .html in a browser. The relative `<video src>` resolves to
`../aria_gen2_videos/<filename>.mp4` (we don't move the MP4s).
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

_DEFAULT_VIDEOS = Path("annotation/aria_gen2_videos")
_DEFAULT_OUT = Path("annotation/contact_sheets")
# Session-name extractor: filenames look like
#   AriaGen2PilotDataset_v1.0_<name>_preview_rgb.mp4
# capture <name> as the contact-sheet basename.
_NAME_RE = re.compile(r"AriaGen2PilotDataset_v[\d.]+_(?P<name>.+?)_preview_rgb\.mp4")


def _video_duration_s(mp4: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(mp4)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _extract_thumb(mp4: Path, t_s: float, out_path: Path, width: int) -> bool:
    """Extract one JPEG at `t_s` via ffmpeg keyframe seek; return True on success."""
    if out_path.exists():
        return True
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{t_s}", "-i", str(mp4),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", "4",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and out_path.exists()


def _render_html(
    title: str,
    mp4_rel: str,
    stride_sec: float,
    thumbs: list[tuple[float, str]],  # (t_sec, rel_jpg_path)
) -> str:
    """Return the full HTML page as a string."""
    cards = "\n".join(
        f"""    <div class="card" data-t="{t:.3f}" onclick="jump({t:.3f})">
      <img src="{html.escape(jpg, quote=True)}" loading="lazy" alt="t={t:.1f}s">
      <div class="ts">t={t:.1f}s</div>
    </div>"""
        for t, jpg in thumbs
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — Aria Gen 2 annotation</title>
<style>
  :root {{
    color-scheme: dark;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font: 14px/1.4 system-ui, -apple-system, sans-serif;
    background: #111;
    color: #eee;
  }}
  header {{
    position: sticky; top: 0; z-index: 10;
    background: #1a1a1a; padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    display: grid; grid-template-columns: minmax(320px, 600px) 1fr; gap: 16px;
    align-items: start;
  }}
  video {{ width: 100%; max-height: 60vh; background: #000; }}
  .ctl {{ display: flex; flex-direction: column; gap: 8px; }}
  .ctl h1 {{ font-size: 16px; margin: 0; color: #f5d76e; }}
  .ctl .meta {{ color: #aaa; font-size: 12px; }}
  .ctl .now {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 18px; padding: 8px 12px; background: #0e0e0e;
    border: 1px solid #333; border-radius: 4px;
    user-select: all;
  }}
  .ctl .yaml {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px; padding: 8px 12px; background: #0e0e0e;
    border: 1px solid #333; border-radius: 4px; white-space: pre;
    user-select: all;
  }}
  .ctl .hint {{ font-size: 11px; color: #888; }}
  main {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
    padding: 16px;
  }}
  .card {{
    cursor: pointer; background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 4px; overflow: hidden;
    transition: border-color 0.1s;
  }}
  .card:hover {{ border-color: #f5d76e; }}
  .card img {{ display: block; width: 100%; height: auto; }}
  .card .ts {{
    font-family: ui-monospace, monospace; font-size: 12px;
    padding: 4px 6px; color: #aaa;
  }}
</style>
</head>
<body>
<header>
  <video id="v" controls preload="metadata" src="{html.escape(mp4_rel, quote=True)}"></video>
  <div class="ctl">
    <h1>{html.escape(title)}</h1>
    <div class="meta">{len(thumbs)} thumbs at stride {stride_sec:.1f}s · click to seek</div>
    <div class="now" id="now">t = 0.000 s</div>
    <div class="yaml" id="yaml">  - id: qN
    query: ""
    intervals:
      - [0.000, 0.000]
    notes: ""</div>
    <div class="hint">YAML scaffold updates on every seek. Edit q-id/query/notes and paste into questions.yaml.</div>
  </div>
</header>
<main>
{cards}
</main>
<script>
const v = document.getElementById("v");
const now = document.getElementById("now");
const yamlBox = document.getElementById("yaml");
function fmt(t) {{ return t.toFixed(3); }}
function jump(t) {{ v.currentTime = t; v.play().catch(()=>{{}}); }}
function refresh() {{
  const t = v.currentTime;
  now.textContent = "t = " + fmt(t) + " s";
  // ±1.5s scaffold matches our atomic_descriptions interval default;
  // adjust by hand if the action span is longer.
  const t0 = Math.max(0, t - 1.5).toFixed(3);
  const t1 = (t + 1.5).toFixed(3);
  yamlBox.textContent =
    "  - id: qN\\n" +
    "    query: \\"\\"\\n" +
    "    intervals:\\n" +
    "      - [" + t0 + ", " + t1 + "]\\n" +
    "    notes: \\"\\"";
}}
v.addEventListener("timeupdate", refresh);
v.addEventListener("seeked", refresh);
refresh();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos", type=Path, default=_DEFAULT_VIDEOS,
        help="Directory containing AriaGen2PilotDataset_v1.0_<name>_preview_rgb.mp4 files.",
    )
    ap.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output directory for one .html + <name>_thumbs/ per session.",
    )
    ap.add_argument(
        "--stride-sec", type=float, default=5.0,
        help="Seconds between thumbnails (default 5; lower = more thumbs).",
    )
    ap.add_argument(
        "--thumb-width", type=int, default=240,
        help="JPEG width in pixels (default 240; height auto from aspect).",
    )
    ap.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated session names to include; default = all found.",
    )
    args = ap.parse_args()

    if not args.videos.is_dir():
        print(f"ERR: {args.videos} not a directory; run "
              "scripts/download_aria_gen2_videos.py first", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only.split(",")) if args.only else None

    mp4s = sorted(args.videos.glob("AriaGen2PilotDataset_v*_preview_rgb.mp4"))
    if not mp4s:
        print(f"ERR: no Aria Gen 2 preview MP4s under {args.videos}", file=sys.stderr)
        return 1

    for mp4 in mp4s:
        m = _NAME_RE.match(mp4.name)
        if m is None:
            print(f"  WARN: unparseable filename {mp4.name}; skipping", file=sys.stderr)
            continue
        name = m.group("name")
        if wanted and name not in wanted:
            continue

        dur = _video_duration_s(mp4)
        if dur <= 0:
            print(f"  WARN: ffprobe returned 0 duration for {mp4.name}; skipping",
                  file=sys.stderr)
            continue

        thumbs_dir = args.out / f"{name}_thumbs"
        thumbs_dir.mkdir(exist_ok=True)

        n_steps = int(dur // args.stride_sec) + 1
        print(f"[contact] {name}: dur={dur:.1f}s -> {n_steps} thumbs "
              f"@ stride {args.stride_sec}s", file=sys.stderr)

        thumbs: list[tuple[float, str]] = []
        for i in range(n_steps):
            t = i * args.stride_sec
            if t > dur:
                break
            jpg = thumbs_dir / f"t{t:08.3f}.jpg"
            if not _extract_thumb(mp4, t, jpg, args.thumb_width):
                print(f"  WARN: ffmpeg failed at t={t:.1f}s for {name}", file=sys.stderr)
                continue
            rel_jpg = f"{name}_thumbs/{jpg.name}"
            thumbs.append((t, rel_jpg))

        # The MP4 lives in `args.videos`, the HTML in `args.out`. Build a
        # relative path from the HTML to the MP4 so the directory is movable.
        try:
            mp4_rel = str(Path(mp4.resolve()).relative_to(args.out.resolve()))
        except ValueError:
            # MP4 isn't under args.out — use a sibling-style ../path.
            import os
            mp4_rel = os.path.relpath(mp4.resolve(), args.out.resolve())

        html_path = args.out / f"{name}.html"
        html_path.write_text(_render_html(
            title=name, mp4_rel=mp4_rel,
            stride_sec=args.stride_sec, thumbs=thumbs,
        ))
        print(f"  wrote {html_path} ({len(thumbs)} thumbs)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
