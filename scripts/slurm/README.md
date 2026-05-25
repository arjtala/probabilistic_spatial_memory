# SLURM job scripts

Batch wrappers for PSM jobs that need cluster resources (GPU extraction,
parallel CPU sweeps). Each script is parameter-free at the call site:
all knobs come from `sbatch --export=KEY=VAL`, with sensible defaults
baked in for the FAIR `/checkpoint/dream/arjangt/video_retrieval/aria`
layout.

## One-time setup on a fresh cluster checkout

The sbatch scripts assume the `psm` conda env exists with the project's
extraction package installed editable + CLIP deps. From a fresh clone:

```bash
# 1. Create + activate the env (or reuse an existing one):
conda create -n psm python=3.12 -y
conda activate psm

# 2. Install the project, CLIP deps, and the YAML loader scripts need:
cd /storage/home/$USER/src/probabilistic_spatial_memory
pip install -e "./extraction[clip]"

# 3. Confirm the imports the baselines need actually resolve:
python -c "import h5py, numpy, yaml; from psm_extraction.models import make_runner"

# 4. Warm the HuggingFace cache for the CLIP checkpoints (compute nodes
#    typically have no internet access; the sbatch jobs run in offline
#    mode and will fail if a model isn't cached).
python -c "
from transformers import CLIPModel, CLIPProcessor
for ckpt in [
    'openai/clip-vit-base-patch32',
    'laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
    'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
]:
    print(f'[hf-warm] {ckpt}')
    CLIPProcessor.from_pretrained(ckpt)
    CLIPModel.from_pretrained(ckpt)
print('[hf-warm] done')
"
```

If step 3 raises `ModuleNotFoundError`, the install didn't take — most
commonly because `pip install` ran against system Python instead of the
conda env's. Verify with `which pip` (should be inside the env dir).

Step 4 downloads ~5.5 GB total (bigG dominates) into
`~/.cache/huggingface/hub/`. Cached weights are shared between the
login and compute nodes, so once is enough.

## Conventions

- **Log destination**: `logs/<job-name>_<jobid>.{out,err}` in the
  submission directory. The scripts `mkdir -p logs` so this works from
  any cwd.
- **Conda env**: every script does `source activate $CONDA_ENV`
  (default `psm`). Override with `--export=CONDA_ENV=myenv`. This is
  the cluster-standard pattern — works in non-interactive batch shells
  without needing conda init.
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

### `eval_psm_sweep.sbatch`

GPU job that runs the PSM evaluation seed sweep (`scripts/eval_bigg_all.sh`
internally) for one or more encoders. Per encoder: 3 sessions × 5 seeds
× ~20 questions = 300 PSM queries per encoder. Produces JSONs at
`$CAPTURES/eval_<sid>_<TAG>_e128_s<seed>.json`. These are the PSM half
of the E11 baselines-vs-PSM comparison.

- Account/QoS: `dream` / `h200_comm_shared` (1× H200)
- Resources: 8 CPUs, 64 GB, 2 hour wall-time cap
- Expected wall-time: ~15 min for 2 encoders, longer if HF cache cold
- Submit: `sbatch scripts/slurm/eval_psm_sweep.sbatch`
- **Prerequisite**: `targets/psm` must be compiled (`make all`). The
  script exits early with a clear error if the binary is missing.

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
# 0. One-time setup (see top of README): conda env + pip install + HF cache warm.

# 1. Compile the C engine (one-time):
make all
test -x targets/psm && echo "psm built"

# 2. Backfill the missing CLIP-L extraction (GPU, ~15 min)
sbatch scripts/slurm/extract_clipL_fulham.sbatch

# 3. Run the PSM seed sweep (GPU, ~15 min)
sbatch scripts/slurm/eval_psm_sweep.sbatch

# 4. Run the baseline sweep (CPU, ~10 min). Can run in parallel with #3
#    since they don't conflict on output paths.
sbatch scripts/slurm/eval_baselines.sbatch

# 5. Wait for both to finish (check with `squeue -u $USER` or
#    `tail -f logs/*_*.out`).

# 6. Aggregate baselines + PSM side-by-side
python scripts/eval_aggregate.py --by-seed --label-from-features \
  captures/eval_*_clipBigG_e128_s*.json \
  captures/baselines/*.json
```

## Troubleshooting

- **Job activation fails because of conda.** The scripts use the
  cluster-standard `source activate <env>` pattern. If the env doesn't
  exist or the name is wrong, the error is immediate. Check with
  `conda env list` on the login node.

- **`httpx.ConnectError` / `Connection reset by peer` mid-run.** The
  compute node has no internet access and a CLIP checkpoint isn't in
  the local HF cache. The sbatch scripts set `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` so this should now fail with a clearer
  `OSError: ... is not a local folder` instead of a network timeout.
  Fix: warm the cache from a login node (see "One-time setup" step 4
  above), then resubmit. If you genuinely need to hit the hub from a
  job, submit with `--export=HF_HUB_OFFLINE=0,TRANSFORMERS_OFFLINE=0`
  — but most compute nodes can't reach huggingface.co anyway.
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
