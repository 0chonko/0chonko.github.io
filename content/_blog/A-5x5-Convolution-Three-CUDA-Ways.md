---
title: "A 5×5 Convolution, Three CUDA Ways"
date: 2026-10-13
description: "Progressive kernel results on a fixed 4096×4096 stencil: naive, then shared tile, then constant-mask broadcast, with GPix/s and the PCIe wall-clock caveat."
---

A 5×5 image stencil looks like a textbook GPU exercise until you ask where the bytes actually come from. On an RTX 3060 Ti (Ampere, `sm_86`, CUDA 13.2), I wanted three CUDA variants on the same problem: global-memory naive, shared-memory tile, shared tile plus **constant memory** for the mask. The interesting part is not that shared memory helps (it usually does), but how much a broadcast-friendly mask buys you once the compare is honest about grid size and normalization.

The upstream repo shipped mismatched shapes (1024 vs 12800). I patched every variant to **4096×4096**, **5×5 filter divided by 35**, block **32×16**, and checked **TEST PASSED** before trusting timings. Hardware: **RTX 3060 Ti**, **sm_86**, **CUDA 13.2**. Three timed runs per kernel; I kept the stable minimum. Output rate is `4096² / t` from kernel time only.

```bash
make clean && make
./convolution   # or project-specific binary after fair-size patch
nvcc --ptxas-options=-v -c convolution_*.cu   # per translation unit
```

---

## Performance for 4096×4096 convolution

Each thread owns one output pixel. A 5×5 stencil reads 25 neighbors per output. Same grid for every step; each step adds one kernel and keeps the earlier bars in view.

### 1. Naive (global)

Start with the textbook kernel: each thread writes one output pixel and pulls all 25 stencil taps straight from global memory, image and mask alike. No staging, no broadcast path. ptxas reports **16** registers, **0** shared, **0** constant. This is the floor you compare everything else against, not a straw man.

| Kernel | Time | Output rate |
|--------|------|-------------|
| Naive | ~1.58 ms | ~10.6 GPix/s |

```mermaid
flowchart LR
  N["Naive · 1.58 ms · 10.6 GPix/s"]
```

On Ampere that number is already not terrible. L2 is catching a lot of the repeated neighbor reads, so the next optimization has less headroom than the textbook slide deck assumes.

### 2. Shared tile

A 32×16 output block with a 5×5 stencil needs a halo. The shared-memory kernel cooperatively loads that tile once into **2880 B** of smem, hits one `__syncthreads`, then computes from the tile. The mask still comes from global. Register pressure jumps to **40**.

The hope is fewer repeated global loads on interior pixels. The reality on this GPU and this grid:

| Kernel | Time | Output rate |
|--------|------|-------------|
| Naive | ~1.58 ms | ~10.6 GPix/s |
| Shared tile | ~1.51 ms | ~11.1 GPix/s |

```mermaid
flowchart LR
  N["Naive · 1.58 ms"] --> S["Shared · 1.51 ms"]
```

About **4%** off naive. Tiling alone is not the story here: you paid in registers and a barrier for a nudge that L2 was already giving you most of. Worth doing as a building block, not as the headline win.

### 3. Shared + constant mask

Keep the tile. Change where the **mask** lives. The 5×5 weights are identical for every thread, so loading them from global on every tap is pure waste: 25 extra global transactions per output that do not scale with image reuse. Put them in **`__constant__`** (100 B, `cmem[3]`) and let the warp broadcast. Registers drop to **36** vs the shared-only path.

| Kernel | Time | Output rate |
|--------|------|-------------|
| Naive | ~1.58 ms | ~10.6 GPix/s |
| Shared tile | ~1.51 ms | ~11.1 GPix/s |
| Shared + constant mask | ~1.15 ms | ~14.6 GPix/s |

```mermaid
flowchart LR
  N["Naive · 1.58 ms"] --> S["Shared · 1.51 ms"]
  S --> C["Shared + cmem · 1.15 ms"]
```

That is the bar that pays: ~**27%** off shared, **1.37×** vs naive. Same stencil math, fair dimensions, two different memory-hierarchy moves stacked: reuse what overlaps in space (image halo), broadcast what repeats across the warp (mask).

The repo also has an **OpenACC** path on **10000×10000**; that is a different toolchain and size, not a fourth CUDA kernel in this A/B.

---

## Caveat: PCIe still owns the clock

Kernel optimization is real but **not** what dominated wall time in this harness. Transfers stay off the ladder:

| Path | Time (approx.) |
|------|----------------|
| Best kernel (shared + cmem) | ~1.15 ms |
| H2D + D2H | ~48–52 ms |
| Wall clock (transfer-heavy) | ~50 ms class |

Report kernel steps first (`TEST PASSED`), then say the end-to-end app is transfer-bound until inputs live on the GPU across iterations.

What I would check next:

- **Nsight Compute (`ncu`)**: DRAM/L2 bytes per output, achieved occupancy vs the 2880 B smem block.
- **Pinned host memory + async copy**: overlap H2D with something useful, or keep the buffer resident and time kernel-only in a loop.

---

## Takeaways

1. **Measure, then name the bottleneck.** Kernel ms looked competitive only after fixing 4096×4096 everywhere; mismatched sizes would have lied about speedup.
2. **Memory hierarchy beats clever arithmetic** on stencils: shared tiles for spatial reuse, constant memory for coefficients that broadcast across a warp.
3. **Fair A/B is part of perf engineering.** Same grid, same normalization (/35), same block shape, pass/fail gate, then min-of-three kernel times; wall clock gets its own line when PCIe is in play.
