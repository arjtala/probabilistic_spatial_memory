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
- **Account / QoS**: scripts default to `--account=dream`. Partition is
  *inferred from the QoS prefix* on this cluster (cpu_* -> cpu,
  h200_* -> h200, h100_* -> h100); passing `--partition` explicitly
  triggers a warning and is ignored. The QoS choices per script:
  - **`cpu_lowest`** (priority 10) — CPU-only jobs. Lowest priority is
    fine for routine work.
  - **`h200_comm_shared`** (priority 5) — shared H200 jobs. Good
    default for one-shot extraction.
  - **`h200_dream_high`** (priority 100) — dedicated dream-team H200.
    Reserve for long-running, time-sensitive work (E10 MLLM serving,
    E5 reranker batches).
  - **`h200_dev`** (priority 100, DenyOnLimit) — strict dev QoS for
    debugging sbatch scripts. Don't use for actual work.

  To check the QoS list for your user/account: `sacctmgr show
  associations user=$USER format=Account,QOS`.

## Available jobs

### `extract_clipL_fulham.sbatch`

One-shot GPU extraction to backfill the missing
`1501677363692556/clip_l_features.h5`. The other two sessions already
have CLIP-L extractions; running this closes the encoder-coverage gap
so the E11 baseline sweep can compare bigG vs CLIP-L across all 3
sessions, not just 2 of 3.

- Account/QoS: `dream` / `h200_comm_shared` (1× H200)
- Resources: 8 CPUs, 64 GB, 1 hour wall-time cap
- Expected wall-time on H200: ~10-15 minutes
- Submit: `sbatch scripts/slurm/extract_clipL_fulham.sbatch`

### `eval_baselines.sbatch`

CPU-only sweep that runs the three E11 retrieval baselines (brute-force
CLIP, sliding-window CLIP, uniform-sample CLIP) across all sessions ×
encoders. Forwards to `scripts/eval_baselines_all.sh`.

- Account/QoS: `dream` / `cpu_lowest` (CPU-only)
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

- **`Run 'conda init' before 'conda activate'`** or **`could not find
  conda.sh`** in the .err log. Means the script couldn't locate your
  conda install in the search path. Fix: find your conda's base dir
  (`echo $CONDA_PREFIX` from an activated shell, then strip the
  trailing `/envs/<env>` if any), and submit with `--export`:

  ```bash
  sbatch --export=ALL,CONDA_BASE=$HOME/miniforge3 \
    scripts/slurm/eval_baselines.sbatch
  ```

  The `ALL,` prefix preserves your other env vars while adding
  CONDA_BASE. If you find yourself overriding it on every submit,
  add a permanent entry to the candidate list inside `activate_conda`
  at the top of each sbatch script.
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
- **`Invalid qos specified`** or similar. Means the QoS doesn't exist
  for your user/account. Run `sacctmgr show associations user=$USER
  format=Account,QOS` to see what's valid, then either edit the
  `#SBATCH --qos=` line or submit with `--qos=<right_qos>`.
- **`sbatch: WARNING: Partition is inferred from QOS prefix so
  --partition flag is ignored`**. Cosmetic — the scripts don't pass
  `--partition` precisely to avoid this warning. If you see it, you've
  probably added `--partition=...` at submit time; drop it.
- **Job pending in queue**. `squeue -u $USER` to see state, `sinfo
  --partition=cpu` (or `h200`) for partition health. The
  `cpu_lowest` / `h200_comm_shared` QoS are intentionally low priority;
  bump to `h200_dream_high` (priority 100) if a job sits longer than
  acceptable.
