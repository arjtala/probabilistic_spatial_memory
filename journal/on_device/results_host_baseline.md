# Host baseline — synthetic engine bench (for host→ARM ratio in §4.3)

Machine: Apple Silicon (macOS arm64), brew libh3 4.4.1.
Binary: core-only `benchmark_spatial_memory`, `-O3` (no `-march=native` needed on Apple clang arm64).
Args: `200000 1024 200000 <cap>` — 5 reps per cap. Median + p95 below.

Note: `peak_VmHWM` is Linux-only, so RSS is a **device-only** number (not available on macOS).
Ingest is cap-independent (per_cell_cap only affects `query_similar`); the small
cap=1 vs cap=10 ingest difference is run-to-run noise.

| Quantity | cap=1 median | cap=1 p95 | cap=10 median | cap=10 p95 |
|---|---|---|---|---|
| Ingest (µs/frame) | 0.329 | 0.333 | 0.336 | 0.343 |
| query_intervals (µs) | 30.195 | 31.435 | 30.551 | 31.540 |
| query_similar (µs) | 366.457 | 378.876 | 548.061 | 559.968 |

**Regime confirmation:** cap=1 (366 µs) is *faster* than cap=10 (548 µs) for
query_similar — matches the guide's regime note (1024 sparse tiles → cap=K's
larger candidate sort dominates). Report on-device *absolute* µs + RSS; do not map
this synthetic cap ordering onto §4.3's Nymeria cap ordering.

Raw stdout: `/tmp/host_baseline_out.txt` (10 runs).
Pending: S22 (Exynos 2200, Cortex-X2 via `taskset 80`) numbers for the same args.
