# Reproducing the Localization Paradox Demo

Companion runbook to `localization_paradox.md`. Everything here is one-time setup + per-session eval. Output JSONs and PNGs land in `captures/`.

---

## Prereqs

```bash
# activate your Python environment with psm-extraction[clip,viz] installed
pip install -e extraction[clip,viz]
```

The `[clip]` extra pulls torch + torchvision + transformers + Pillow. The `[viz]` extra pulls matplotlib + umap-learn + h3 (used by `scripts/embedding_atlas.py` and `scripts/compose_screenshots.py`).

---

## 1. Extract CLIP-L features per session

~3 minutes per 15-min session on M4 Pro MPS.

```bash
for sid in 1501677363692556 287142033569927; do
  python -m psm_extraction extract \
    --video  datasets/$sid/data.mp4 \
    --output datasets/$sid/clip_l_features.h5 \
    --models clip \
    --checkpoint clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --sample-fps 3 --segment-sec 1 \
    --session-id $sid
done

# Session C uses a different file layout: video.mp4 (not data.mp4) and
# no JSON sidecars; GPS lives inside the original features.h5.
python -m psm_extraction extract \
  --video       datasets/201703061033/video.mp4 \
  --output      datasets/201703061033/clip_l_features.h5 \
  --models      clip \
  --checkpoint  clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
  --sample-fps  3 --segment-sec 1 \
  --gps-source  datasets/201703061033/features.h5 \
  --session-id  201703061033
```

For DINO/JEPA overlays in `psm-viz`, swap `--models clip` for `--models dino,jepa`. JEPA is the long pole (~10× DINO's per-frame cost).

---

## 2. Run eval per session, 5 seeds each

15 runs total at ~30 s each.

```bash
for sid in 1501677363692556 287142033569927 201703061033; do
  for seed in 0 1 2 3 4; do
    python scripts/eval_lookback.py \
      datasets/$sid/clip_l_features.h5 \
      datasets/$sid/questions.yaml \
      --top 5 --time-window 75 --capacity 12 --exemplars 128 \
      --clip-checkpoint laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
      --seed $seed \
      --out captures/eval_${sid}_clipL_e128_s${seed}.json
  done
done
```

Each `questions.yaml` lives next to its session's video and lists scored queries (with annotated `intervals`) plus negative controls (with `intervals: []`). `session_start_unix` is auto-detected from `clip/timestamps[0]` if omitted.

---

## 3. Aggregate across sessions and seeds

```bash
python scripts/eval_aggregate.py --by-seed --label-from-features \
  captures/eval_*_clipL_e128_s*.json \
  --out captures/eval_seedsweep_clipL_e128.json
```

Prints a per-session row + combined row with mean ± std on `exemplar mIoU @1/@k`, `exemplar Hit @k`, and `bucket mIoU @k`. Emits a JSON with the same data plus per-seed breakdown for downstream slicing.

---

## 4. Companion figures

**Embedding atlas — paired view** (geographic × embedding correspondence):

```bash
python scripts/embedding_atlas.py \
  datasets/1501677363692556/clip_l_features.h5 \
  --method umap --mode paired \
  --query bus=datasets/q-bus.bin \
  --query zebra=datasets/q-zebra.bin \
  --out captures/embedding_atlas_paired.png
```

**Embedding atlas — similarity grid** (per-query gradient over the cloud):

```bash
python scripts/embedding_atlas.py \
  datasets/1501677363692556/clip_l_features.h5 \
  --method umap \
  --query bus=datasets/q-bus.bin \
  --query zebra=datasets/q-zebra.bin \
  --query river=/tmp/psm-e5/river/query.f32 \
  --out captures/embedding_atlas_grid.png
```

**Three-session screenshot strip** (after capturing one PNG per session via `psm-viz` `P`):

```bash
python scripts/compose_screenshots.py --orient vertical \
  --label "Fulham, London (A) — Aria, walking" \
  --label "Palo Alto, CA (C) — Honda HDD, driving" \
  --label "Tucson, AZ (B) — Aria, cycling" \
  --out captures/three_sessions_strip.png \
  datasets/1501677363692556/captures/psm-viz-000000.png \
  datasets/201703061033/captures/psm-viz-000000.png \
  datasets/287142033569927/captures/psm-viz-000000.png
```

---

## 5. Visual proof in `psm-viz`

To verify any returned `(cell, exemplar_t)` against the actual video:

```bash
targets/psm-viz -d datasets/<session_id> -g dino   # uses features.h5 for the attention overlay
```

In the visualizer:
- `Space` to start playback (it launches paused on the first frame).
- Scroll horizontally on the video pane to scrub the timeline; jump to the `exemplar_t` reported in the eval JSON.
- `M` cycles heatmap modes (`total` → `current` → `recency`).
- `E` toggles 3D hex extrusion.
- `P` saves a screenshot to `<session_dir>/captures/`.

---

## Reproducibility checklist

Record alongside any captured artifact:

- `git rev-parse HEAD`
- Build profile (`local` / `portable` / `debug` / `sanitize`)
- Extraction checkpoint (e.g. `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`)
- `psm` flags: `--top`, `--exemplars`, `--k-ring`, `-r`, `-t`, `-C`, `-p`, `-g`, `--seed`
- Per top-1 hit: query path, `cell`, `(lat, lng)`, `similarity`, `exemplar_t`, `(t_min, t_max)`

Apples-to-apples comparisons across encoder / reservoir / retention configurations are then a matter of changing one knob at a time and re-running the loop in §2.
