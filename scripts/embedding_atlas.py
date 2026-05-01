#!/usr/bin/env python3
"""Project per-frame embeddings to 2D and render an inferno-density atlas.

Pairs with psm-viz's geographic heatmap to show what spatial memory
*contains* (semantic neighborhoods) alongside where it *landed*
(H3 cells over the route).

Usage:
    python scripts/embedding_atlas.py datasets/1501677363692556/clip_features.h5

    # Overlay top-k matches for one or more text queries:
    python scripts/embedding_atlas.py datasets/1501677363692556/clip_features.h5 \\
        --query bus=datasets/q-bus.bin \\
        --query zebra=datasets/q-zebra.bin \\
        --query river=/tmp/psm-e5/river/query.f32 \\
        --out captures/embedding_atlas.png

    # Force the deterministic PCA fallback when UMAP isn't installed:
    python scripts/embedding_atlas.py FEATURES.h5 --method pca

UMAP gives better cluster structure; PCA is deterministic and dependency-free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def load_embeddings(path: Path, group: str):
    with h5py.File(path, "r") as f:
        if group not in f:
            raise SystemExit(f"group {group!r} not in {path}; have {list(f.keys())}")
        emb = f[f"{group}/embeddings"][:].astype(np.float32)
        ts = f[f"{group}/timestamps"][:].astype(np.float64)
        lat = f[f"{group}/lat"][:].astype(np.float64)
        lng = f[f"{group}/lng"][:].astype(np.float64)
    return emb, ts, lat, lng


def project(emb: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    if method == "pca":
        # Center and use truncated SVD (top-2 components).
        centered = emb - emb.mean(axis=0, keepdims=True)
        # Use partial SVD via numpy; for ~3k frames this is fine.
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
        return coords.astype(np.float32)
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "umap not installed. Either `pip install umap-learn` or pass --method pca."
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
        )
        return reducer.fit_transform(emb).astype(np.float32)
    raise SystemExit(f"unknown method: {method}")


def load_query_vector(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise SystemExit(f"{path}: size {len(raw)} not a multiple of 4 (float32)")
    return np.frombuffer(raw, dtype=np.float32).copy()


def parse_query_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = Path(spec).stem, spec
        out[label] = Path(path)
    return out


def compute_h3_cells(lat: np.ndarray, lng: np.ndarray, resolution: int) -> np.ndarray:
    try:
        import h3  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "h3 python binding required for --mode paired. "
            "pip install h3"
        ) from exc
    # h3 v4: latLngToCell; h3 v3: geo_to_h3. Probe.
    if hasattr(h3, "latlng_to_cell"):
        encode = lambda la, ln: h3.latlng_to_cell(float(la), float(ln), resolution)
    elif hasattr(h3, "geo_to_h3"):
        encode = lambda la, ln: h3.geo_to_h3(float(la), float(ln), resolution)
    else:
        raise SystemExit("unrecognized h3 python API (need latlng_to_cell or geo_to_h3)")
    cells = np.empty(lat.shape[0], dtype=object)
    for i in range(lat.shape[0]):
        cells[i] = encode(lat[i], lng[i])
    return cells


def _style_axes(ax, title: str) -> None:
    ax.set_facecolor("black")
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def render_density(coords: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="black")
    nbins = 96
    hist, xe, ye = np.histogram2d(coords[:, 0], coords[:, 1], bins=nbins)
    ax.imshow(
        np.log1p(hist).T,
        origin="lower",
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        cmap="inferno",
        aspect="auto",
        alpha=0.95,
        interpolation="bilinear",
    )
    ax.scatter(
        coords[:, 0], coords[:, 1], s=1.5, c="white", alpha=0.12, linewidths=0
    )
    _style_axes(ax, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[atlas] wrote {out_path}", file=sys.stderr)


def render_paired(
    lat: np.ndarray,
    lng: np.ndarray,
    coords: np.ndarray,
    cells: np.ndarray,
    emb: np.ndarray,
    queries: dict[str, Path],
    out_path: Path,
    title: str,
    topk: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stable per-cell color from a discrete palette. tab20 covers 20 cells;
    # the included session has ~10 cells, so collisions are unlikely.
    unique_cells = sorted(set(cells.tolist()))
    cmap_disc = matplotlib.colormaps["tab20"]
    cell_color = {c: cmap_disc(i % 20) for i, c in enumerate(unique_cells)}
    colors = np.array([cell_color[c] for c in cells])

    fig, (ax_geo, ax_emb) = plt.subplots(
        1, 2, figsize=(15.0, 7.5), facecolor="black"
    )

    # Geographic panel: lng on x, lat on y, equal-area aspect via cos(lat).
    mid_lat = float(np.median(lat))
    cos_lat = np.cos(np.deg2rad(mid_lat))
    ax_geo.scatter(lng, lat, c=colors, s=10, alpha=0.85, linewidths=0)
    ax_geo.set_xlabel("lng (°)", color="white")
    ax_geo.set_ylabel("lat (°)", color="white")
    ax_geo.set_aspect(1.0 / max(cos_lat, 1e-3))

    # Embedding panel: UMAP/PCA 2D coords colored by the same cell.
    ax_emb.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, alpha=0.85, linewidths=0)
    ax_emb.set_xlabel("dim 1", color="white")
    ax_emb.set_ylabel("dim 2", color="white")

    # Query overlays as white-edged circles in BOTH panels.
    if queries:
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        palette = ["#FFFFFF", "#FFD166", "#7DF9FF", "#FF6EC7", "#A0E060"]
        for i, (label, qpath) in enumerate(queries.items()):
            q = load_query_vector(qpath)
            if q.shape[0] != emb.shape[1]:
                raise SystemExit(
                    f"query {label} dim {q.shape[0]} != embeddings dim {emb.shape[1]}"
                )
            q = q / (np.linalg.norm(q) + 1e-9)
            sims = emb_norm @ q
            idx = np.argsort(-sims)[:topk]
            edge = palette[i % len(palette)]
            ax_geo.scatter(
                lng[idx], lat[idx],
                s=44, facecolors="none", edgecolors=edge, linewidths=1.0,
                label=f"{label}", zorder=4,
            )
            ax_emb.scatter(
                coords[idx, 0], coords[idx, 1],
                s=44, facecolors="none", edgecolors=edge, linewidths=1.0,
                label=f"{label}", zorder=4,
            )

    # Per-panel styling.
    _style_axes(ax_geo, "geographic (lat / lng)")
    _style_axes(ax_emb, "embedding (UMAP)")

    if queries:
        leg = ax_emb.legend(
            loc="best", fontsize=8, framealpha=0.7,
            facecolor="#111111", edgecolor="#444444",
        )
        for txt in leg.get_texts():
            txt.set_color("white")

    # Cell-color legend below both panels: short H3 prefix per swatch.
    swatch_w = 0.8 / max(len(unique_cells), 1)
    for i, c in enumerate(unique_cells):
        rect_x = 0.1 + i * swatch_w
        fig.patches.append(
            plt.Rectangle(
                (rect_x, 0.02), swatch_w * 0.9, 0.018,
                transform=fig.transFigure,
                facecolor=cell_color[c], edgecolor="none",
            )
        )
        fig.text(
            rect_x + swatch_w * 0.45, 0.005,
            c[-6:],  # last 6 hex chars are visually distinct enough
            transform=fig.transFigure,
            color="white", fontsize=6, ha="center",
        )

    fig.suptitle(title, color="white", fontsize=12, y=0.985)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[atlas] wrote {out_path}", file=sys.stderr)


def render_similarity_grid(
    coords: np.ndarray,
    emb: np.ndarray,
    queries: dict[str, Path],
    out_path: Path,
    title: str,
    topk: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(queries)
    if n == 0:
        raise SystemExit("similarity-grid mode needs at least one --query")
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 5.0 * rows), facecolor="black", squeeze=False
    )

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    # Symmetric vmin/vmax across panels so palette is comparable.
    sims_per_q: dict[str, np.ndarray] = {}
    for label, qpath in queries.items():
        q = load_query_vector(qpath)
        if q.shape[0] != emb.shape[1]:
            raise SystemExit(
                f"query {label} dim {q.shape[0]} != embeddings dim {emb.shape[1]}"
            )
        q = q / (np.linalg.norm(q) + 1e-9)
        sims_per_q[label] = emb_norm @ q
    all_sims = np.concatenate(list(sims_per_q.values()))
    vmin = float(np.percentile(all_sims, 1))
    vmax = float(np.percentile(all_sims, 99))

    for ax, (label, sims) in zip(axes.flat, sims_per_q.items()):
        # Plot in ascending similarity so high-similarity dots draw last (on top).
        order = np.argsort(sims)
        ax.scatter(
            coords[order, 0],
            coords[order, 1],
            c=sims[order],
            cmap="inferno",
            s=6,
            alpha=0.85,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        # Top-k as small white-edged dots so the "winning" frames are
        # legible against the gradient.
        idx = np.argsort(-sims)[:topk]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=22,
            facecolors="none",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        _style_axes(ax, f"{label}  (max sim {sims.max():.3f})")

    # Hide unused subplots if any.
    for ax in axes.flat[len(sims_per_q):]:
        ax.set_visible(False)

    fig.suptitle(title, color="white", fontsize=12, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[atlas] wrote {out_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path, help="HDF5 features file")
    ap.add_argument("--group", default="clip", help="HDF5 group (default: clip)")
    ap.add_argument(
        "--method",
        default="umap",
        choices=("umap", "pca"),
        help="2D projection method (default: umap; falls back to pca on import error)",
    )
    ap.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="overlay top-k matches for a query.f32 file; repeatable",
    )
    ap.add_argument("--topk", type=int, default=30, help="frames outlined per panel in similarity mode")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "density", "similarity-grid", "paired"),
        help=(
            "auto: similarity-grid if --query given else density. "
            "paired: 2-panel geographic + embedding, both colored by H3 cell. "
            "Optional --query overlays draw on both panels."
        ),
    )
    ap.add_argument(
        "--h3-resolution",
        type=int,
        default=10,
        help="H3 resolution for --mode paired (default 10, matches psm default)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("captures/embedding_atlas.png"),
        help="output PNG path",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb, ts, lat, lng = load_embeddings(args.features, args.group)
    print(
        f"[atlas] {args.features.name}::{args.group}: "
        f"{emb.shape[0]} frames, dim={emb.shape[1]}, "
        f"duration={ts[-1] - ts[0]:.1f}s",
        file=sys.stderr,
    )

    print(f"[atlas] projecting via {args.method}...", file=sys.stderr)
    coords = project(emb, args.method, seed=args.seed)

    queries = parse_query_specs(args.query)
    title = (
        f"{args.features.name} :: {args.group}  "
        f"({emb.shape[0]} frames, {args.method})"
    )

    mode = args.mode
    if mode == "auto":
        mode = "similarity-grid" if queries else "density"

    if mode == "density":
        render_density(coords, args.out, title)
    elif mode == "similarity-grid":
        render_similarity_grid(coords, emb, queries, args.out, title, args.topk)
    elif mode == "paired":
        cells = compute_h3_cells(lat, lng, args.h3_resolution)
        n_unique = len(set(cells.tolist()))
        print(
            f"[atlas] {n_unique} H3 cells at resolution {args.h3_resolution}",
            file=sys.stderr,
        )
        render_paired(
            lat, lng, coords, cells, emb, queries, args.out, title, args.topk
        )
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
