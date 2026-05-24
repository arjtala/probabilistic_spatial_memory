# SLURM job scripts

Batch wrappers for PSM jobs that need cluster resources (GPU extraction,
parallel CPU sweeps). Each script is parameter-free at the call site:
all knobs come from `sbatch --export=KEY=VAL`, with sensible defaults
baked in for the FAIR `/checkpoint/dream/arjangt/video_retrieval/aria`
layout.

## Conventions

- **Log destination**: `logs/<job-name>_<jobid>.{out,err}` in the
  submission directory. The scripts `mkdir -p logs` so this works from
  any cwd.
- **Conda env**: every script activates `$CONDA_ENV` (default `psm`)
  via `conda activate` after sourcing `conda.sh`. Override with
  `--export=CONDA_ENV=myenv` if needed.
- **Data root**: `$ROOT` (default `/checkpoint/dream/arjangt/video_retrieval/aria`).
  Override for local runs with `--export=ROOT=$PWD/datasets`.
- **Partition / QoS**: defaults to `learnfair`. If your cluster uses a
  different partition (`fairaws`, `devlab`, etc.), edit the `#SBATCH
  --partition=` line at the top or override at submit time:
  `sbatch --partition=fairaws --export=... <script>`.

## Available jobs

### `extract_clipL_fulham.sbatch`

One-shot GPU extraction to backfill the missing
`1501677363692556/clip_l_features.h5`. The other two sessions already
have CLIP-L extractions; running this closes the encoder-coverage gap
so the E11 baseline sweep can compare bigG vs CLIP-L across all 3
sessions, not just 2 of 3.

- Resources: 1 GPU, 8 CPUs, 64 GB, 1 hour wall-time cap
- Expected wall-time on H200: ~10-15 minutes
- Submit: `sbatch scripts/slurm/extract_clipL_fulham.sbatch`

### `eval_baselines.sbatch`

CPU-only sweep that runs the three E11 retrieval baselines (brute-force
CLIP, sliding-window CLIP, uniform-sample CLIP) across all sessions ×
encoders. Forwards to `scripts/eval_baselines_all.sh`.

- Resources: 4 CPUs, 16 GB, 30 min wall-time cap (overkill — the sweep
  finishes in 5-10 min on a single core, the headroom is for first-run
  HuggingFace text-encoder downloads)
- Submit: `sbatch scripts/slurm/eval_baselines.sbatch`

## Workflow for the first paper-grade E11 pass

```bash
# 1. Backfill the missing CLIP-L extraction (GPU, ~15 min)
sbatch scripts/slurm/extract_clipL_fulham.sbatch

# 2. Wait for it to finish (check with `squeue -u $USER` or
#    `tail -f logs/extract_clipl_fulham_*.out`).

# 3. Run the baseline sweep (CPU, ~10 min)
sbatch scripts/slurm/eval_baselines.sbatch

# 4. Inspect the output JSONs once the job lands
ls captures/baselines/

# 5. Aggregate baselines + PSM side-by-side
python scripts/eval_aggregate.py --by-seed --label-from-features \
  captures/eval_*_clipBigG_e128_s*.json \
  captures/baselines/*.json
```

## Troubleshooting

- **`conda: command not found` in the log**. The script tries to source
  `conda.sh` from `$(conda info --base)`, but if `conda` isn't on the
  batch shell's PATH, sourcing fails before activation. Fix: prepend
  your conda's bin to `PATH` in `~/.bashrc`, or hardcode the source
  line at the top of the sbatch script to your conda install (e.g.
  `source /private/home/$USER/miniconda3/etc/profile.d/conda.sh`).
- **`features.h5` not found**. The baseline sweep skips missing files
  with a WARN; it doesn't fail. If you see WARN lines for a session,
  either the extraction never ran or the path/basename is wrong. Check
  `ls $ROOT/<sid>/` against the encoder→basename mapping in
  `scripts/eval_baselines_all.sh`.
- **Out of GPU memory during extraction**. Lower the CLIPPyTorchRunner
  batch size by editing
  `extraction/psm_extraction/models/clip_pytorch.py` (default is 16).
  H200 has 141 GB so this shouldn't bite, but if it does on a smaller
  GPU, the symptom is a CUDA OOM ~15s into the run.
- **Job pending in queue**. `squeue -u $USER` to see the state, `sinfo`
  for partition health. `learnfair` typically schedules within minutes;
  if it's been > 15 min check the partition is up.
