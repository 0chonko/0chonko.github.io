---
title: "One Heat Stencil, Four Runtimes"
date: 2026-10-13
description: "Progressive results on a 1000×1000 heat stencil: sequential baseline, AVX2, then pthreads and OpenMP thread counts, with wall time and FLOP/s."
---

I keep a small **8-neighbor heat stencil** around as a sanity check: conductivity-weighted update with
`DIRECT = (√2/(√2+1))/4` on cardinal neighbors and `DIAG = (1/(√2+1))/4` on diagonals, double-buffered temperatures, fixed iteration count. Same math, four runtimes (seq, hand SIMD, OpenMP, pthreads). On this box the surprise was not SIMD (expected win) but **OpenMP losing to pthreads** once threads actually help.

Fixed size throughout: **1000×1000**, **2000** iterations. Each step below adds one result and keeps the earlier ones in the same table.

## Setup

| Item | Value |
|------|--------|
| Grid | 1000×1000, 2000 iterations |
| Input | `areas_1000x1000.pgm`, tolerance `e=1e-4` |
| Compiler | gcc 15.3, `-O3 -march=native` (all variants) |
| OpenMP env | `OMP_PROC_BIND=true`, `OMP_WAIT_POLICY=active` |

SIMD path: `immintrin.h`, **`__m256d`**, `VEC_LENGTH=4`, **`ALIGN=64`**. OpenMP: `#pragma omp for nowait` on the interior, **`critical`** when reporting global min/max; team fork/join **every outer iteration**. Pthreads: barrier between halo and interior phases, **`mutex`** on the reduction.

---

## Performance for 1000×1000 heat stencil

Times and FLOP/s are program-reported from the benchmark stdout.

### 1. Sequential baseline

Plain scalar loop, no threads.

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |

```mermaid
flowchart LR
  S["seq · 4.882 s · 4.92e9 FLOP/s"]
```

### 2. Vectorize the hot loop (AVX2)

Hand SIMD on the interior (`__m256d`, `VEC_LENGTH=4`). Same single-threaded control flow:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| simd AVX2 | 2.772 | 8.66e9 |

```mermaid
flowchart LR
  S["seq · 4.882 s"] --> V["AVX2 · 2.772 s"]
```

About **1.76×** vs seq with no threads. Min/max matched the sequential reference; the printed **`T_diff`** did not (order **6.7** vs ~**0.0047**). Classic reduction bug on that one aggregate. I still trust the timing loop.

### 3. Pthreads: raise the thread count

Persistent worker threads, barrier between halo and interior. Start at one thread (runtime tax, no parallel payoff), then 8, then 16:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| simd AVX2 | 2.772 | 8.66e9 |
| pth -p1 | 5.859 | 4.10e9 |

`-p1` is already slower than seq. Add parallel work:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| simd AVX2 | 2.772 | 8.66e9 |
| pth -p1 | 5.859 | 4.10e9 |
| pth -p8 | 2.190 | 1.10e10 |

Eight threads is the peak so far (~**2.2×** vs seq). Push to 16:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| simd AVX2 | 2.772 | 8.66e9 |
| pth -p1 | 5.859 | 4.10e9 |
| pth -p8 | 2.190 | 1.10e10 |
| pth -p16 | 3.110 | 7.72e9 |

```mermaid
flowchart LR
  P1["pth-p1 · 5.859 s"] --> P8["pth-p8 · 2.190 s"]
  P8 --> P16["pth-p16 · 3.110 s"]
```

Sixteen goes backwards: still faster than `-p1`, slower than `-p8`. Contended bandwidth, false sharing on halos, barrier skew.

### 4. OpenMP: same thread counts, different runtime

Team fork/join **every** outer iteration. Same size, same iteration count:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| omp -p1 | 6.988 | 3.43e9 |

Worse than seq with no parallel work. Add threads:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| omp -p1 | 6.988 | 3.43e9 |
| omp -p8 | 4.583 | 5.24e9 |

Barely clears seq. Then 16:

| Variant | Time (s) | FLOP/s |
|---------|----------|--------|
| seq | 4.882 | 4.92e9 |
| omp -p1 | 6.988 | 3.43e9 |
| omp -p8 | 4.583 | 5.24e9 |
| omp -p16 | 6.803 | 3.53e9 |

```mermaid
flowchart LR
  O1["omp-p1 · 6.988 s"] --> O8["omp-p8 · 4.583 s"]
  O8 --> O16["omp-p16 · 6.803 s"]
```

`-p16` regresses below seq again. Best overall across the whole shootout remains **pthreads at 8 threads**.

---

## Why OpenMP lost to pthreads

The dominant cost on this micro-benchmark is **fork/join tax**. Creating or waking an OpenMP team **2000 times** (once per time step) buys little when each step is already cache-friendly work split across rows. Pthreads stay alive; they hit a **barrier** per iteration but skip repeated team setup.

At 8 threads, pthreads overlap interior rows cleanly after the halo barrier. OpenMP's `-p8` only nudges past seq because parallel row work and **`nowait`** help, but not enough to catch AVX2 or pth-8.

---

## Connection

The stencil is memory-bound at this grid size; the ranking is mostly **how much synchronization you pay per iteration**, not FLOP count. That rhymes with playing cache hierarchy in simulation ([Simulating a cache with SystemC](/blog/Simulating-a-cache-with-SystemC/)): here the "model" is the runtime, and the expensive events are barriers and team lifecycle, not the multiply-adds.

---

## Takeaways

1. **Vectorize the hot loop first** on CPU: AVX2 cut time ~43% with no threads.
2. **Measure at the thread count that matches the machine**, not "more is better": 16 threads hurt OpenMP; pth peaked at 8.
3. **OpenMP inside a time-step loop** needs a persistent parallel region or `omp parallel` around the whole iteration nest; per-step fork/join ate the win here.
4. **Validate reductions separately** from timings; a wrong `T_diff` would have sent me on a wild goose chase.
5. For fair runtime shootouts, report **wall time and a derived FLOP/s** together; my OpenMP FLOP/s at `-p1` looks artificially low because time blew up on overhead.

Anyway, next step on this problem is one OpenMP variant with a **single parallel region** over all 2000 steps and a proper parallel reduction for diff, then rerun the same progressive tables.
