#!/usr/bin/env python3
"""Stack screenshots into a single vertical or horizontal strip.

Use:

    python scripts/compose_screenshots.py \\
        --orient vertical \\
        --label "Fulham, London" --label "Palo Alto, CA" --label "Tucson, AZ" \\
        --out captures/sessions_strip.png \\
        datasets/1501677363692556/captures/psm-viz-000000.png \\
        datasets/201703061033/captures/psm-viz-000000.png \\
        datasets/287142033569927/captures/psm-viz-000000.png

Inputs are normalized to a common width (vertical) or height (horizontal),
preserving aspect, and stacked on a configurable background. Optional
per-input labels are drawn as captions above each pane.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    """Try a handful of system fonts; fall back to PIL's default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _resize_to_width(img: Image.Image, w: int) -> Image.Image:
    h = round(img.height * w / img.width)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def _resize_to_height(img: Image.Image, h: int) -> Image.Image:
    w = round(img.width * h / img.height)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument(
        "--orient",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="vertical = stack tall; horizontal = stack wide",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        help="per-input caption; repeat once per input. Skipped if empty.",
    )
    ap.add_argument(
        "--bg", default="black", help="background color (PIL color string)"
    )
    ap.add_argument(
        "--label-color", default="white", help="caption text color"
    )
    ap.add_argument(
        "--gap", type=int, default=12,
        help="pixels of background between panes",
    )
    ap.add_argument(
        "--label-height", type=int, default=44,
        help="caption strip height in pixels",
    )
    ap.add_argument(
        "--font-size", type=int, default=22,
        help="caption font size",
    )
    args = ap.parse_args()

    if args.label and len(args.label) != len(args.inputs):
        raise SystemExit(
            f"--label given {len(args.label)} time(s) but {len(args.inputs)} input(s)"
        )

    images = [Image.open(p).convert("RGB") for p in args.inputs]
    font = _load_font(args.font_size) if args.label else None

    if args.orient == "vertical":
        target_w = max(im.width for im in images)
        normalized = [_resize_to_width(im, target_w) for im in images]
        per_pane_h = [
            (im.height + (args.label_height if args.label else 0))
            for im in normalized
        ]
        total_h = sum(per_pane_h) + args.gap * (len(images) - 1)
        canvas = Image.new("RGB", (target_w, total_h), args.bg)
        draw = ImageDraw.Draw(canvas) if font else None
        y = 0
        for idx, im in enumerate(normalized):
            if args.label and draw is not None:
                label = args.label[idx]
                # Caption text vertically centered in the label band.
                bbox = draw.textbbox((0, 0), label, font=font)
                text_h = bbox[3] - bbox[1]
                ty = y + (args.label_height - text_h) // 2 - bbox[1]
                draw.text((16, ty), label, fill=args.label_color, font=font)
                y += args.label_height
            canvas.paste(im, (0, y))
            y += im.height
            if idx + 1 < len(images):
                y += args.gap
    else:
        target_h = max(im.height for im in images)
        normalized = [_resize_to_height(im, target_h) for im in images]
        total_w = sum(im.width for im in normalized) + args.gap * (len(images) - 1)
        label_band_h = args.label_height if args.label else 0
        canvas = Image.new("RGB", (total_w, target_h + label_band_h), args.bg)
        draw = ImageDraw.Draw(canvas) if font else None
        x = 0
        for idx, im in enumerate(normalized):
            canvas.paste(im, (x, label_band_h))
            if args.label and draw is not None:
                label = args.label[idx]
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                tx = x + (im.width - text_w) // 2 - bbox[0]
                ty = (label_band_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
                draw.text((tx, ty), label, fill=args.label_color, font=font)
            x += im.width
            if idx + 1 < len(images):
                x += args.gap

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out} ({canvas.width}×{canvas.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
