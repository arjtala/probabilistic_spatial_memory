# Analyst brief — figures/tables for the PSM negative-result paper (branch: revision-prep)

Produce figures/tables + a point-in-time analysis from the **frozen** artifacts. Every
number must trace to a committed JSON/manifest. **Do NOT touch the 60-drive cohort banks**
(unfrozen/unused — keep them so). Coordinate with the writer (two-findings structure).

### Deliverables
1. **Finding 1 — fixed-budget retention** (`section_5_results.tex tab:fixed-budget`):
   6 policies at M=128 (+ M=64 secondary), Hit@5 / mIoU / oracle / rare-place / common-place,
   with paired session-bootstrap CIs. Headline: `spatial_priority` flat vs `global_reservoir`;
   `semantic_kcenter` best.
2. **Finding 2 — viability gate (the "not constructible" figure):** CLIP-L vs **SigLIP2**
   matched arm, prompts v1/v2/v3, top-5 rate vs the 50% bar, + degeneracy-guard cosines.
   Source: `pilot/viability_v{1,2,3}.json` + `_siglip2.json`. Make the encoder-robustness
   explicit (v3 = 25.0% under both, different subsets).
3. **60-drive cohort characterization:** durations, revisited-place counts, coordinate
   provenance, coverage — from `hdd_cohort_v1_manifest.json`.
4. **Coordinate postmortem (methods warning):** the silent-clamp displacement
   (median/p95; the ~5–65 km collapse vs the corrected ~10–62 m) and the guard that caught
   it. Note `docs/HDD.md:264` F-HDD-2/3 are **VOID** — do not reproduce them.

### Rules
- Run whatever cluster analysis is needed against the frozen artifacts (the user
  facilitates cluster access). Nothing new run against the cohort banks.
- All figures reproducible from committed artifacts + `scripts/{caption_viability.py,
  eval_fixed_budget.py, compare_fixed_budget.py}`.
- Follow the dataviz house style; keep metric names identical to the submitted paper.

### Cluster environment (read before the first run — each of these cost a cycle)
- `PSM_DATA_ROOT` is **unset**; the data root is `/checkpoint/dream/arjangt`. SLURM
  scripts referencing it fail instantly with `PSM_DATA_ROOT: unbound variable` unless
  `ROOT`/`OUT_ROOT` are passed via `--export=ALL,...`.
- Use `/home/arjangt/.conda/envs/psm/bin/python`; the default pod python lacks
  `yaml`/`h5py`. `ffmpeg`/`ffprobe` are in that env but **not on `PATH`**.
- `HF_HUB_OFFLINE=1` is mandatory for CLIP-L / SigLIP2 loads (the proxy 403s
  HuggingFace). Both checkpoints are already cached under `~/.cache/huggingface/hub`.
- **No GitHub egress from the sandbox pod** (SSH and HTTPS both blocked). Commits land in
  the shared working tree; pushes come from the login node. A failing `git fetch` means
  blocked egress, not a missing commit — verify with
  `git merge-base --is-ancestor <sha> HEAD`, not `ls-remote`.
- GPU: `--qos=h200_dev` is what actually schedules. The 3-drive SigLIP2 array took ~6 min
  per task.

### Do not mix encoders (silent-subset hazard)
`hdd/features/*/*/` now holds **two** feature files per drive:
- `clip_l_features.h5` — all **132** drives, coordinates corrected at rest, carries the
  `gps_realign_fix` root attr. **This is the canonical file for everything.**
- `siglip2_l_features.h5` — only drives **3 / 47 / 131** (`201702271438`,
  `201704111540`, `201710061345`), extracted solely for the Amendment C viability arm.

Globbing `*_features.h5` will silently mix encoders across an incomplete subset. Match
`clip_l_features.h5` explicitly, and use the SigLIP2 three only for the Finding-2
matched-arm figure.
