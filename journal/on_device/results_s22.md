# S22 on-device results — synthetic engine bench

**Device:** Galaxy S22 global (SM-S901B), **Exynos 2200**, aarch64.
**Core:** pinned to CPU 7 = **Cortex-X2** prime via `taskset 80` (verified: `ps` PSR=7).
**Governor:** `energy_aware` (non-root; no `performance` pin available — caveat below).
**Binary:** core-only `benchmark_spatial_memory`, cross-compiled with NDK r29
(`aarch64-linux-android34-clang`, clang 21), `-O3 -mcpu=cortex-x2`, **static libh3
4.5.0** (aarch64), bionic-only deps. Ran fully over adb; no on-device build.
**Args:** `200000 1024 200000 <cap>`. 10 reps/cap; **rep 1 dropped as warmup**, n=9.

## Three numbers (Reviewer 2)

| Quantity | cap=1 median (p95) | cap=10 median (p95) |
|---|---|---|
| **Ingest** | 0.864 µs/frame (0.898) | 0.874 µs/frame (1.093) |
| **Query (location, `query_intervals`)** | 104.3 µs (106.3) | 104.6 µs (106.1) |
| **Query (semantic, `query_similar`)** | 1163 µs (1203) | 1746 µs (1792) |
| **Peak RSS** | 20.2 MiB (max 20.3) | 20.3 MiB (max 20.4) |

## Host → ARM ratio (S22 median / host median, same synthetic bench)

| Metric | cap=1 | cap=10 |
|---|---|---|
| Ingest | 2.63× | 2.60× |
| query_intervals | 3.45× | 3.42× |
| query_similar | 3.17× | 3.19× |

Host baseline: Apple Silicon macOS arm64, brew h3 4.4.1 (see
`results_host_baseline.md`). Consistent ~2.6–3.5× host→wearable slowdown.

## Notes
- **cap ordering:** cap=1 (1163 µs) < cap=10 (1746 µs) for `query_similar` —
  same direction as host (366 < 548 µs), matching the guide's regime note
  (1024 sparse tiles → cap=K's larger candidate sort dominates). Do **not** map
  this synthetic cap split onto §4.3's Nymeria cap ordering.
- **No thermal throttling:** cap=1 query_similar flat at 1154–1211 µs across
  reps 2–10; cap=10 flat at 1719–1800 µs. No upward drift → burst == sustained
  for these ~4–6 min runs.
- **query_intervals rep-1 warmup effect:** cap=1 rep1=61.7 µs, rep2=84.6 µs,
  then steady ~104 µs from rep3 on — first two reps warm before the plateau;
  median over reps 2–10 = 104.3 µs is the steady-state value.
- **RSS** rock-steady ~20.2–20.4 MiB at the 1024-tile / 200k-op state — a few MiB,
  consistent with the O(area) ~0.44 MiB/cell bound framing.
- **Non-root caveats:** `energy_aware` governor (no `performance` pin); numbers are
  big-core (X2) burst under charger. `taskset 80` affinity confirmed via PSR=7.

Raw stdout: `/tmp/s22_out.txt` (20 runs) + `/tmp/host_baseline_out.txt` (10 runs).
