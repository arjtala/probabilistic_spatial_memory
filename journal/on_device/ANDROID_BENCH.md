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
