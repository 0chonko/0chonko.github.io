---
title: "OpenMP Offload on AMD GPUs: Notes from an HLRS Tutorial"
date: 2025-10-22
description: "What I took away from HLRS's two-day course on OpenMP target offload, unified shared memory on MI300A, discrete-GPU data environments, and ROCm profiling. [Certificate](/assets/blog/openmp-offload-hlrs-tutorial/personal-certificate.pdf)"
---

Most of my GPU work so far has been CUDA-centric: kernels, NVLink bandwidth, power profiling on A100 and H100 nodes. OpenMP on CPUs I know from cache-line tuning in a SystemC simulator and from MPI+OpenMP hybrid jobs on Snellius. But **OpenMP target offload** on AMD hardware was a gap, and it kept showing up in the same conversations as portability and Fortran weather codes.

I spent two days in an online tutorial at **HLRS** on **OpenMP offloading with AMD GPUs**, working through exercises on the **AMD Accelerator Cloud (AAC)** against **MI300A** APUs. The course walked from "add a `#pragma omp target` and hope" to understanding when unified shared memory actually removes copies, when you need explicit `map` clauses on discrete GPUs, and how to read `LIBOMPTARGET_KERNEL_TRACE` before reaching for `rocprofv3`. These are my condensed notes, plus where I think this connects to the MPI and profiling work I have been doing elsewhere.

---

## Why Another GPU Programming Model?

CUDA and HIP give you fine control. OpenMP's pitch is different: keep a **standard directive-based model** that composes with host-side threading and ports across vendors. On AMD, the stack is **ROCm + LLVM-based `amdclang`/`amdflang`**, with target offload wired into the same compiler you use for host OpenMP.

The tutorial framed the decision less as ideology and more as economics:

* **APUs (MI300A)** share one physical memory pool between CPU and GPU. Porting a CPU loop can be almost mechanical if you accept unified shared memory semantics.
* **Discrete GPUs (MI300X and friends)** still need explicit data environments. The directives look similar; the runtime behavior is not.
* **Fortran production codes** (weather, climate, chemistry) are already OpenMP-heavy on the host. Target offload is the path of least resistance compared to rewriting hot loops in HIP.

That last point landed when Johanna Potyka walked through **CLOUDSC** and other ECMWF-style mini-apps: the question is not "can OpenMP express this kernel?" but "how far does `target teams distribute parallel do collapse(...)` get you before compiler quirks and data-movement costs bite?"

---

## Hardware Context: MI300A as an APU, Not a Bolt-On GPU

Before any pragmas, the course spent time on architecture taxonomy. An **APU** is not just a GPU glued to a CPU socket. On MI300A, CPU chiplets and CDNA3 compute dies share **128 GB HBM3** in a unified address space. No separate host DRAM and device HBM to shuttle between over PCIe.

```text
Discrete GPU (MI300X)          APU (MI300A)
+------------------+           +---------------------------+
| Host DRAM        |           | Unified HBM3 (128 GB)     |
|       | PCIe     |           |  CPU cores + 228 CUs      |
|       v          |           |  same physical pages      |
| GPU HBM          |           +---------------------------+
+------------------+
```

For porting, that split matters immediately:

| Mode | Typical hardware | Data movement | What you add in source |
| :--- | :--- | :--- | :--- |
| Discrete (default) | MI300X, MI250 | Explicit H2D/D2H copies via `map` | `map(to:...)` / `map(from:...)` |
| USM on discrete | MI300X + page migration | Zero-copy via XNACK | `#pragma omp requires unified_shared_memory` + `HSA_XNACK=1` |
| APU zero-copy | MI300A | No redundant copies in the common case | USM pragma or `-fopenmp-force-usm` |

The exercises ran on **aac6** (open ROCm 6.4.1) and **aac7** (HPE Cray Programming Environment on production MI300A). With 50+ participants, the SLURM etiquette was part of the lesson: `salloc` with `--gpus=1` and `--mem` capped below a full node so four people could share one machine.

---

## Day 1: Unified Shared Memory and the First Surprise

### USM: pointers that mean the same thing everywhere

**Unified Shared Memory (USM)** is an OpenMP contract, not just a ROCm feature. You declare it once per translation unit:

```cpp
#pragma omp requires unified_shared_memory
```

On AMD, you also need **XNACK** enabled at compile time (`--offload-arch=gfx942`) and runtime:

```bash
export HSA_XNACK=1
```

With USM active, `map` clauses become optional. Host pointers passed into `target` regions can trigger **page migration** on discrete hardware, or direct access on MI300A without the copy dance.

The mental model from Michael Klemm's slides: without USM you wrap compute in a data environment that uploads at the opening brace and downloads at the close. With USM you can often write:

```cpp
#pragma omp requires unified_shared_memory

double *in  = (double*)malloc(M * sizeof(double));
double *out = (double*)malloc(M * sizeof(double));

// host initialization writes visible to the device after migration
for (int i = 0; i < M; i++) in[i] = ...;

#pragma omp target teams distribute parallel for
for (int i = 0; i < M; i++)
  out[i] = f(in[i]);

// host reads out[] without an explicit map(from:)
```

That is the fastest on-ramp for porting CPU OpenMP loops. The course was explicit that it is not free performance: migration has its own cost, and mixing USM with `map` on the same buffers changes access semantics to coarse grain.

### Offload is not parallelism (and my first broken saxpy)

This was the first debugging cliff in the exercises. A lone `#pragma omp target` transfers control to the device **sequentially and synchronously**. It does not create a GPU parallel team. You still need `teams distribute parallel for` (or the Fortran `do` equivalent).

Worse, my first Fortran saxpy "worked" (checksum matched) but `LIBOMPTARGET_KERNEL_TRACE=1` showed **1 team x 256 threads** and miserable occupancy. The kernel ran on the GPU; it just did not use it.

MI300A exposes **228 CUs** with **64-wide wavefronts**. OpenMP cannot easily span the entire device with a flat `parallel for` the way you might imagine from CPU threading. The fix is multi-level parallelism:

```cpp
#pragma omp target teams distribute parallel for
for (int i = 0; i < n; i++)
  y[i] = a * x[i] + y[i];
```

On the GPU, `teams` maps to workgroups, `distribute` tiles the iteration space across them, and `parallel for` fills each CU with wavefronts. AMD's compilers ignore host `simd` on device code anyway; the hierarchy is the whole story.

### Discrete memory: when USM is off, maps are the contract

The second lecture block switched to **discrete-GPU semantics** even on AAC hardware, by setting `HSA_XNACK=0` and using explicit maps.

A minimal saxpy with tracing told the story better than the slides alone. Running the same kernel twice with `LIBOMPTARGET_KERNEL_TRACE=2`:

```text
Call data_submit_async: 450us  ... 4194304 bytes
Call launch_kernel:       43us
Call data_retrieve_async: 204us ... 4194304 bytes
Call data_delete:         99us
```

Four megabytes each way, twice, for two back-to-back target invocations. The kernel itself was **43 microseconds**. This is the discrete-GPU trap in one log line: you can "successfully offload" and still spend two orders of magnitude more time moving data than computing.

The fix is not a fancier pragma on a single call. It is **persistent data environments**:

```cpp
#pragma omp target data map(to: a[:N], b[:N]) map(tofrom: c[:N])
{
  compute_kernel_1(...);  // nested targets find data via presence checks
  saxpy(...);
  compute_kernel_2(...);
}
```

Or the unscoped variant for longer-lived state:

```fortran
!$omp target enter data map(to: input(1:N)) map(alloc: tmp(1:N))
! ... multiple target regions ...
!$omp target exit data map(from: res) map(delete: tmp)
```

`target update` bridges host-side changes into an existing device environment without tearing the whole map down. That pattern is what separates a demo from something that could survive a time step loop.

---

## Day 2: Real Codes, Real Pitfalls

### Fortran defaults will race on you

The application-examples session opened with a trap I would have walked into. In Fortran:

```fortran
real(kind=rt) :: tmp = 0.0_rt   ! initialized declaration
real(kind=rt) :: a(10000)
!$omp target teams distribute parallel do
do i = 1, 10000
  tmp = real(i, kind=rt)
  a(i) = tmp
end do
```

Output flickers every run. Why? An initialized local variable is a **SAVE** variable in Fortran, which OpenMP treats like a **shared** module variable. Every iteration races on `tmp`. Remove the initialization (making `tmp` a true local) or use `default(none)` with explicit `private(tmp)`.

The course recommendation was blunt: on device code, **`default(none)`** unless you have a good reason not to. Same spirit as verifying UCX transport on Snellius: assumptions that hold on CPU OpenMP silently break on GPU.

### SPMD mode and not splitting your pragmas

Another compiler-facing detail: **`target teams distribute parallel for` on one line** (Fortran: continuation with `&`) encourages **SPMD** code generation. Splitting:

```fortran
!$omp target
!$omp teams distribute parallel do
```

can produce different outlining and optimization paths depending on the compiler. Not always wrong, but the course treated gratuitous separation as a portability smell.

### Weather mini-apps: collapse is not infinite

CLOUDSC and related kernels stress **column-parallel weather physics**: lots of independent columns, deep loop nests, Fortran array syntax. The porting question becomes how aggressively `collapse` can fuse loop dimensions before register pressure and cache behavior on CDNA3 push back.

Practical guidance from the examples:

* Port **compute-heavy loops** first; leave `workshare` and array-syntax assignments for later.
* OpenMP 6.0 introduces `workdistribute` for whole-array operations, but **no compiler in the room implemented it yet**. The workaround is a `distribute parallel do` over indices.
* **`allocate` inside a target loop** is fragile (broken on CCE 18 in one example; possible with `amdflang` and `-lflang_rt.hostdevice`, but slow). Hoist allocations outside the kernel.

### Jacobi and OpenFOAM-style porting

The slides also walked Jacobi solvers and CFD mini-apps (OpenFOAM-related examples in the materials), emphasizing the same loop: get correctness with USM on APU, then turn off XNACK and rebuild data environments for discrete-GPU performance, then profile before micro-optimizing launch parameters.

---

## Profiling: Start Cheap, Then Open the ROCm Toolbox

Luka Stanisic's profiling block mirrored how I approach Snellius jobs: **free introspection first**, heavy tools only when you know what question you are asking.

### Layer 0: runtime printf from libomptarget

Before `rocprofv3`:

```bash
export LIBOMPTARGET_KERNEL_TRACE=1   # kernel names, teams, threads, registers
export LIBOMPTARGET_KERNEL_TRACE=2   # adds timing for kernels and data transfers
export LIBOMPTARGET_INFO=$((0x20 | 0x02 | 0x01 | 0x10))  # verbose data movement
```

On the saxpy exercise, level 2 was enough to prove the bottleneck was `data_submit_async`, not kernel launch.

### Layer 1: rocprofv3 for timelines and hotspots

AMD's current tracing CLI, built on **rocprofiler-sdk**:

```bash
rocprofv3 --kernel-trace --stats --summary -- ./my_openmp_app
rocprofv3 --sys-trace --output-format pftrace -- ./my_openmp_app
```

For OpenMP offload, `--kernel-trace` (or `--sys-trace`) captures GPU dispatches initiated by target regions. It does **not** profile host-side OpenMP parallel regions, which is a gap to remember when you have hybrid host/device parallelism.

Open the `.pftrace` in [Perfetto](https://ui.perfetto.dev/) and you can correlate H2D copies, kernel boxes, and HSA activity on one timeline. The course exercise pointed at `HPCTrainingExamples/Rocprofv3/OpenMP/`.

### Layer 2: rocprof-compute for kernel autopsy

When you know *which* `__omp_offloading_*` kernel hurts, **rocprof-compute** (formerly Omniperf) replays the app to collect hardware counters and roofline data:

```bash
rocprof-compute profile -n my_kernel -- ./my_openmp_app
rocprof-compute analyze -p workloads/my_kernel/<gpu_arch>/
```

OpenMP outlining can obscure source names in profiles; the tool keys off **dispatch IDs** and launch geometry, which still expose whether you launched one team or hundreds.

### Layer 3: rocprof-sys for whole-program context

**rocprof-sys** (formerly Omnitrace) is the choice when you need host threads, GPU activity, and eventually MPI in one trace. Overkill for a saxpy exercise; relevant when OpenMP offload sits inside a multi-rank job.

| Tool | Best for | OpenMP offload caveat |
| :--- | :--- | :--- |
| `LIBOMPTARGET_*` env vars | Quick copy vs kernel sanity check | Text only, but zero setup |
| `rocprofv3` | Timelines, hotspots, CSV counters | GPU side of offload; not host `parallel` |
| `rocprof-compute` | Per-kernel roofline and stalls | Multiple replays; kernel names may be mangled |
| `rocprof-sys` | End-to-end application trace | Heavier; OMPT offload support evolving |

---

## What This Connects To (and What I Have Not Tried)

This course sits next to several threads from my own work without replacing any of them:

* **CUDA energy profiling** taught me to separate measurement overhead from the phenomenon. `LIBOMPTARGET_KERNEL_TRACE=2` is the same instinct for offload: read the transfer lines before optimizing kernel tile sizes.
* **UCX/MPI on Snellius** was about proving the runtime took the fast path. OpenMP offload has an analogue: USM with `HSA_XNACK=1` vs explicit maps with `HSA_XNACK=0` are different machines. Mixing them without noticing is the silent fallback.
* **SystemC OpenMP cache tuning** was host-only, but the Fortran `SAVE` variable pitfall is the same class of bug: language semantics you internalized on CPU do not evaporate on GPU.
* **GASPI/HPC communication** is farther up the stack, yet the Jacobi and ghost-exchange examples in the broader HLRS materials hint at where GPU-aware MPI and OpenMP offload eventually meet.

What I have not validated myself: whether a production weather kernel ported with USM on MI300A, then tightened with `target data` for MI300X, beats a hand-tuned HIP port on the same science case. And I have not run `rocprofv3` on an OpenMP+MPI job at scale. Those feel like the natural follow-ups, not another pass through the slides.

---

## Practical Takeaways

1. **Treat `target` and `parallel` as separate decisions.** Offload moves execution; `teams distribute parallel for` creates GPU parallelism. A correct checksum with 1x256 threads is a failed port.
2. **Pick your memory model deliberately.** USM plus `HSA_XNACK=1` for fast CPU-to-GPU migration on APUs; explicit `target data` / `enter data` when XNACK is off or copies must be minimized.
3. **Read `LIBOMPTARGET_KERNEL_TRACE=2` before rocprof.** If `data_submit_async` dominates `launch_kernel`, fix data persistence before touching tile sizes.
4. **Use `default(none)` on device loops.** Fortran initialized locals are `SAVE` and therefore shared. This will race.
5. **Keep `target teams distribute` fused when you can.** SPMD outlining is compiler-sensitive; don't split pragmas without a measured reason.
6. **Profile GPU dispatches with `rocprofv3 --kernel-trace`.** Host OpenMP and device offload are different subsystems; the tool only sees one of them.
7. **Verify on your architecture flag.** `--offload-arch=gfx942` for MI300A in AAC; discrete MI300X is the same ISA but a different memory story at runtime.

---

### References

* HLRS / AMD, *Introduction to OpenMP Offloading with AMD GPUs* (2-day online course, Oct 21 to 22, 2025). [Course materials](https://fs.hlrs.de/projects/par/events/2025/GPU-OPENMP-AMD/). [Certificate](/assets/blog/openmp-offload-hlrs-tutorial/personal-certificate.pdf).
* AMD, [HPCTrainingExamples](https://github.com/AMD/HPCTrainingExamples) (OpenMP C/Fortran exercises, Rocprofv3 OpenMP labs).
* AMD ROCm, [OpenMP support and USM](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/openmp.html).
* AMD ROCm, [Using rocprofv3 with OpenMP](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-with-openmp.html).
* Full list: [All references](/readinglist/).

---

