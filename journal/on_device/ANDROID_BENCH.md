# On-device PSM benchmark — Android POC guide

Goal: replace "on-device timing unmeasured" (§5/§6) with a real ARM64 latency +
memory number, addressing the reviewers' #1 gap. Post-submission-safe (camera-ready
or poster) — do NOT block the Jul 25 submission on it.

## What to run
The self-contained synthetic benchmark **`benchmark_spatial_memory`** (the
"1024-tile synthetic stress test" already cited in §4.3). It needs **no data,
no HDF5, no ffmpeg, no CLIP encoder** — it exercises the pure engine core
(H3 lookup + HLL hash + reservoir update + cosine top-K), so the Android number
is directly comparable to the host-CPU constants already in §4.3.

This isolates PSM's on-device cost from the encoder (CLIP-L is heavy on phones;
out of scope for the POC — pre-computed embeddings / synthetic vectors suffice).

## Dependencies (core-only build)
- **Needed:** `libh3` (Uber H3, portable C — cross-compiles for aarch64).
- **Excluded on purpose:** HDF5 and ffmpeg. The default `make` links `-lhdf5`
  via `src/ingest/ingest.o` inside `libpsm.a`; a core-only build skips
  `src/ingest` entirely so HDF5 is never referenced.

Core-only source set:
```
src/core/*.c
vendor/probabilistic_data_structures/lib/{hash,bitarray,utilities}.c
vendor/probabilistic_data_structures/hyperloglog/hll.c
vendor/probabilistic_data_structures/bloom_filter/bloom.c
benchmarks/benchmark_spatial_memory.c
```
Compile with: `-O3 -Iinclude -I. -I<h3>/include ... -lh3 -lm`
(drop `-march=native`; use `-mcpu=native` on-device or a generic aarch64 target).

## Path A — Termux (fastest; native on-device build)
```
pkg install clang make git time
# build libh3 for the phone (cmake) OR pkg install if available, then:
CORE=$(ls src/core/*.c) \
VEND="vendor/probabilistic_data_structures/lib/hash.c \
      vendor/probabilistic_data_structures/lib/bitarray.c \
      vendor/probabilistic_data_structures/lib/utilities.c \
      vendor/probabilistic_data_structures/hyperloglog/hll.c \
      vendor/probabilistic_data_structures/bloom_filter/bloom.c"
clang -O3 -mcpu=native -Iinclude -I. -I$PREFIX/include \
  $CORE $VEND benchmarks/benchmark_spatial_memory.c \
  -L$PREFIX/lib -lh3 -lm -o bench_psm
taskset 80 command time -v ./bench_psm    # pin a big core; -v gives Max RSS
```

## Path B — Android NDK (reproducible; from the dev host)
```
export NDK=$ANDROID_NDK_HOME
CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang
# cross-compile libh3 for aarch64 first (cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a)
$CC -O3 -mcpu=cortex-a715 -Iinclude -I. -I<h3-aarch64>/include \
  src/core/*.c <vendor srcs> benchmarks/benchmark_spatial_memory.c \
  -L<h3-aarch64>/lib -lh3 -lm -o bench_psm
adb push bench_psm /data/local/tmp/ && adb shell 'cd /data/local/tmp && ./bench_psm'
```

## Three numbers to report (Reviewer 2)
1. **Ingest** — µs/frame (or frames/s) into the index.
2. **Query latency** — µs per retrieval at cap=1 and cap=K.
3. **Peak RSS** — `Max resident set size` from `time -v`, or `VmHWM` in
   `/proc/self/status`, vs the ~0.44 MiB/cell O(area) bound.

Pin the performance governor / a big core; run N iterations; report median + p95;
note sustained-vs-burst (thermal) if the numbers drift.

## Paper integration (I'll do this once you send the numbers)
Add a short paragraph + one row to §4.3 "Memory, latency, and failure modes":

> On a <phone, SoC> (aarch64, single big core), the engine ingests at
> <X> µs/frame and answers queries in <Y>/<Z> µs at cap=1/K with <M> MiB peak
> RSS at 10 cells — confirming the microsecond host-CPU constants (§4.3) hold on
> wearable-class ARM. This covers the engine core; CLIP encoding and video I/O
> are excluded (as on host).

Send me the raw numbers (or the `time -v` + bench stdout) and I insert the
verified values — no placeholders ship in the PDF.

---

# Galaxy S22 walkthrough (reviewed target)

**Device:** Galaxy S22 (SM-S901) — Snapdragon 8 Gen 1 (US/CN) or Exynos 2200
(global); both are aarch64 with a **Cortex-X2 prime core = CPU 7**, so
`taskset 80` (hex mask `0x80` = CPU 7) pins the big core on either variant.
Non-rooted is fine — everything below runs in Termux without root.

## Benchmark changes backing this guide (review these first)
`benchmarks/benchmark_spatial_memory.c` gained two small, default-preserving
additions so the phone run emits all three numbers directly:
- **Optional 4th CLI arg `per_cell_cap`** (default `1`) — run once at `1` and
  once at `K` to get the cap=1/K query-latency split §4.3 reports. The
  `query_similar` line now prints `per_cell_cap=<n>`.
- **`peak_VmHWM:` line at exit** (Linux/Android only) — peak RSS straight from
  `/proc/self/status`, so RSS needs no GNU `time`. No-op off Linux.

Default `./bench_psm` (no args) behaves exactly as before.
**Note:** these edits are compile-unverified in the dev clone (no `libh3`
there); first S22 build is the confirmation.

## Step 1 — Termux setup + cross-build libh3 (the one real dependency)
h3 is *not* vendored, so build it once for the phone (installs headers at
`$PREFIX/include/h3/h3api.h`, matching the `#include <h3/h3api.h>` in the core):
```
pkg update && pkg install -y clang make cmake git time util-linux
git clone --depth 1 https://github.com/uber/h3
cd h3 && cmake -B build -DCMAKE_INSTALL_PREFIX=$PREFIX \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_FILTERS=OFF \
  && cmake --build build -j && cmake --install build && cd ..
```

## Step 2 — build the core-only benchmark (on-device)
```
cd <this repo on the phone>
CORE=$(ls src/core/*.c)
VEND="vendor/probabilistic_data_structures/lib/hash.c \
      vendor/probabilistic_data_structures/lib/bitarray.c \
      vendor/probabilistic_data_structures/lib/utilities.c \
      vendor/probabilistic_data_structures/hyperloglog/hll.c \
      vendor/probabilistic_data_structures/bloom_filter/bloom.c"
clang -O3 -mcpu=native -Iinclude -I. -I$PREFIX/include \
  $CORE $VEND benchmarks/benchmark_spatial_memory.c \
  -L$PREFIX/lib -lh3 -lm -o bench_psm
```
(If Termux clang rejects `-mcpu=native`, use `-mcpu=cortex-x2`.)

## Step 3 — run, pinned to the X2 prime core
```
# ingest + query at cap=1 (first run also warms caches; discard run #1)
taskset 80 ./bench_psm 200000 1024 200000 1
# query at cap=K (unrestricted; exemplars/cell = 4, so any cap >= 4, e.g. 10)
taskset 80 ./bench_psm 200000 1024 200000 10
```
Run each ~10×, drop the first as warmup, report **median + p95**.

## Step 4 — read the three numbers off stdout
| Paper quantity | Line to read | Convert |
|---|---|---|
| **Ingest** | `observe_grid ... ops/sec=<X>` | µs/frame = `1e6 / X` |
| **Query cap=1** | run #1: `query_similar ... per_cell_cap=1 ... mean_us=<Y>` | µs as-is |
| **Query cap=K** | run #2: `query_similar ... per_cell_cap=10 ... mean_us=<Z>` | µs as-is |
| **Peak RSS** | `peak_VmHWM:  <M> kB` | MiB = `M/1024` |

Sanity (not a target): cap=1 query > cap=K query (the merge-window-per-unique-
cell effect, §4.3); ingest in the µs/frame range; RSS a few MiB.

## Non-root caveats (report these alongside the numbers)
- **No governor pinning** (root-only). Mitigate: charger in, airplane mode, cool
  room, close background apps, short bursts. If `mean_us` climbs across the 10
  runs, that's thermal throttling — note sustained-vs-burst.
- `taskset` (from `util-linux`) sets affinity without root; `0x80` = CPU 7 =
  Cortex-X2 on both S22 SoCs.

## Host baseline (for the arm64-vs-host comparison in §4.3)
Same benchmark on your dev host (needs the h3 dev package there), so §4.3 can
say "µs on host → µs on wearable-class ARM" for the *same* synthetic workload:
```
CFL=$(pkg-config --cflags h3); LIB=$(pkg-config --libs h3)
clang -O3 -march=native -Iinclude -I. $CFL \
  $CORE $VEND benchmarks/benchmark_spatial_memory.c $LIB -lm -o bench_psm_host
./bench_psm_host 200000 1024 200000 1 ; ./bench_psm_host 200000 1024 200000 10
```

Send me both stdouts (S22 + host) and I fill the §4.3 paragraph + row with the
verified numbers.

