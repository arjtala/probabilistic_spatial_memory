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
Compile with: `-O3 -Iinclude -I. -I<h3>/include $VINC ... -lh3 -lm`
(drop `-march=native`; use `-mcpu=native` on-device or a generic aarch64 target).

**Vendor include dirs are required** — the vendored sources include each other by
bare name (`hll.c` does `#include "hash.h"`), so the compile *fails* without them.
The Makefile passes these via `VENDOR_INCLUDES`; the standalone commands below
must too. Define once and reuse:
```
VINC="-Ivendor/probabilistic_data_structures/lib \
      -Ivendor/probabilistic_data_structures/hyperloglog \
      -Ivendor/probabilistic_data_structures/bloom_filter"
```

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
VINC="-Ivendor/probabilistic_data_structures/lib \
      -Ivendor/probabilistic_data_structures/hyperloglog \
      -Ivendor/probabilistic_data_structures/bloom_filter"
clang -O3 -mcpu=native -Iinclude -I. -I$PREFIX/include $VINC \
  $CORE $VEND benchmarks/benchmark_spatial_memory.c \
  -L$PREFIX/lib -lh3 -lm -o bench_psm
taskset 80 command time -v ./bench_psm    # pin a big core; -v gives Max RSS
```

## Path B — Android NDK (reproducible; from the dev host)
```
export NDK=$ANDROID_NDK_HOME
CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang
# cross-compile libh3 for aarch64 first (cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a)
VINC="-Ivendor/probabilistic_data_structures/lib \
      -Ivendor/probabilistic_data_structures/hyperloglog \
      -Ivendor/probabilistic_data_structures/bloom_filter"
$CC -O3 -mcpu=cortex-a715 -Iinclude -I. -I<h3-aarch64>/include $VINC \
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

## RESULTS — measured on Galaxy S22 (2026-07-10)
Ran end-to-end via **Path B** (NDK r29 cross-compile, static libh3 4.5.0, pushed
over adb; no on-device build). Device: **SM-S901B, Exynos 2200**, pinned to
**CPU 7 = Cortex-X2** via `taskset 80` (affinity confirmed: `ps` PSR=7). Args
`200000 1024 200000 <cap>`, 10 reps/cap, rep 1 dropped as warmup (n=9), median (p95).

| Quantity | cap=1 | cap=10 | host (Apple Silicon) | host→ARM |
|---|---|---|---|---|
| **Ingest** | 0.86 µs/frame (0.90) | 0.87 µs/frame (1.09) | 0.33 µs/frame | ~2.6× |
| **Query — location** (`query_intervals`) | 104 µs (106) | 105 µs (106) | 30 µs | ~3.4× |
| **Query — semantic** (`query_similar`) | 1163 µs (1203) | 1746 µs (1792) | 366 / 548 µs | ~3.2× |
| **Peak RSS** | 20.2 MiB | 20.3 MiB | (Linux-only) | — |

No thermal throttling (query_similar flat across reps 2–10). cap=1 < cap=10 as
predicted by the regime note (matches host direction). Full breakdown +
caveats: **`results_s22.md`** (device) and **`results_host_baseline.md`** (host);
raw stdouts in `/tmp/s22_out.txt`, `/tmp/host_baseline_out.txt`.

## Paper integration — ready-to-insert §4.3 line (verified numbers, no placeholders)
Add after the host-CPU latency sentence in §4.3 "Memory, latency, and failure modes":

> On a Galaxy S22 (Exynos 2200, aarch64, Cortex-X2 prime core), the same synthetic
> engine benchmark ingests at 0.86\,µs/frame and answers a location query in
> 104\,µs and a semantic query in 1.16/1.75\,ms (\caponek{}$=1$/$K$, 1024-tile
> state) with 20.2\,MiB peak RSS — a 2.6--3.4$\times$ slowdown vs.\ the identical
> host benchmark, confirming the microsecond host-CPU constants carry to
> wearable-class ARM at sub-2\,ms, tens-of-MiB cost. This covers the engine core;
> CLIP encoding and video I/O are excluded (as on host). The synthetic cap
> ordering (\caponek{}$=1$ faster) reflects the 1024-sparse-tile workload, not
> §4.3's dense Nymeria regime — see the regime note above.

This also lets the four "on-device timing unmeasured" hedges (§1, §4.3, §6, §7)
be softened to "measured on a Galaxy S22" for a camera-ready / poster. Kept as a
reviewable paper edit — **do NOT block the Jul 25 submission** on it.

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
VINC="-Ivendor/probabilistic_data_structures/lib \
      -Ivendor/probabilistic_data_structures/hyperloglog \
      -Ivendor/probabilistic_data_structures/bloom_filter"
clang -O3 -mcpu=native -Iinclude -I. -I$PREFIX/include $VINC \
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

Sanity (not a target): ingest in the µs/frame range; query µs-scale; RSS a few MiB.

**Regime note (verified by compiling + running this bench, 2026-07-10).** Do NOT
expect §4.3's cap ordering here. §4.3's Nymeria numbers come from a
dense/few-cell workload (~7--10 cells x ~113 exemplars), where the per-cell
merge-window pass dominates and cap=1 is *slower*. This synthetic bench scans
1024 *sparse* tiles x 4 exemplars each, so cap=K's larger candidate sort
dominates and **cap=1 is the FASTER setting**. Confirmed on both machines at the
paper's 200k/1024/200k: host 366 us (cap=1) vs 548 us (cap=10); S22 1163 us vs
1746 us. Both are correct -- different workloads. So: report the on-device
*absolute* query us + RSS as the wearable-feasibility point; do not map the
synthetic cap=1/cap=K split onto §4.3's Nymeria cap ordering.

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
VINC="-Ivendor/probabilistic_data_structures/lib \
      -Ivendor/probabilistic_data_structures/hyperloglog \
      -Ivendor/probabilistic_data_structures/bloom_filter"
clang -O3 -march=native -Iinclude -I. $CFL $VINC \
  $CORE $VEND benchmarks/benchmark_spatial_memory.c $LIB -lm -o bench_psm_host
./bench_psm_host 200000 1024 200000 1 ; ./bench_psm_host 200000 1024 200000 10
```

Send me both stdouts (S22 + host) and I fill the §4.3 paragraph + row with the
verified numbers.

---

# As-executed reproduction (2026-07-10, Path B) — exact commands used

The walkthrough above uses `<h3-aarch64>` placeholders; these are the concrete
commands that produced the results table. Host: macOS arm64 (Apple Silicon).

```
# 1. NDK (host had none; brew cask, run from a real shell — /opt is sandboxed here)
brew install --cask android-ndk        # -> NDK r29 at /opt/homebrew/share/android-ndk
NDK=/opt/homebrew/share/android-ndk
CC=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android34-clang

# 2. cross-compile libh3 (static) for arm64-v8a
git clone --depth 1 https://github.com/uber/h3 /tmp/h3_build/src   # h3 4.5.0
cmake -S /tmp/h3_build/src -B /tmp/h3_build/src/build \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_FILTERS=OFF \
  -DCMAKE_INSTALL_PREFIX=/tmp/h3_build/install
cmake --build /tmp/h3_build/src/build --target h3 -j && cmake --install /tmp/h3_build/src/build
H3=/tmp/h3_build/install     # -> lib/libh3.a (elf64-littleaarch64) + include/h3/h3api.h

# 3. cross-compile the benchmark (static libh3 -> only bionic libc/libm/libdl needed)
VEND="vendor/probabilistic_data_structures/lib/hash.c \
      vendor/probabilistic_data_structures/lib/bitarray.c \
      vendor/probabilistic_data_structures/lib/utilities.c \
      vendor/probabilistic_data_structures/hyperloglog/hll.c \
      vendor/probabilistic_data_structures/bloom_filter/bloom.c"
VINC="-Ivendor/probabilistic_data_structures/lib \
      -Ivendor/probabilistic_data_structures/hyperloglog \
      -Ivendor/probabilistic_data_structures/bloom_filter"
$CC -O3 -mcpu=cortex-x2 -Iinclude -I. -I$H3/include $VINC \
  src/core/*.c $VEND benchmarks/benchmark_spatial_memory.c \
  $H3/lib/libh3.a -lm -o bench_psm_arm64

# 4. push + run, pinned to CPU 7 (Cortex-X2); 10 reps/cap, drop rep 1
adb push bench_psm_arm64 /data/local/tmp/bench_psm && adb shell chmod 755 /data/local/tmp/bench_psm
for cap in 1 10; do for r in $(seq 1 10); do
  adb shell "cd /data/local/tmp && taskset 80 ./bench_psm 200000 1024 200000 $cap"
done; done
# verify it ran on the phone's big core: adb shell ps -A -o PID,PSR,NAME | grep bench_psm  (PSR=7)
```

Gotchas hit (all fixed in the commands above):
- `brew install` into `/opt` fails from a sandboxed tool shell ("Operation not
  permitted" despite correct ownership) — run it from a plain terminal.
- The compile **requires `$VINC`** (vendored sources `#include "hash.h"` etc.
  by bare name) — omitting it fails with `'hash.h' file not found`.
- The S22 (SM-S901B) is the **Exynos 2200** variant; `taskset 80` still pins the
  Cortex-X2 (CPU 7, part `0xd48`) — same as the Snapdragon variant.

