---
title: Experiences with UCX on the Dutch National Supercomputer
date: 2025-06-11
description: "Navigating the complexities of UCX transport selection, unprivileged container namespaces, and network performance debugging on the Snellius supercomputer."
---

CPU and GPU overheads in HPC containers are mostly solved. The network is not: how do containerized message-passing runtimes talk to proprietary network fabrics?

For my Master's thesis at Vrije Universiteit Amsterdam, I characterized the performance and energy of **Snellius**, the Dutch National Supercomputer run by SURF [6, 15]. A large part of that was wrestling with **Unified Communication X (UCX)**, the middleware between MPI runtimes (like OpenMPI) and fast interconnects (InfiniBand HDR/NDR and NVLink).

This is the post I wish I had read before my first benchmark looked fine but ran over TCP: the debugging steps, the namespace flags, and the configs that got UCX working with Apptainer on Snellius CPU and GPU partitions.

---

## 1. Hardware

Two generations of AMD EPYC CPU nodes and two generations of NVIDIA GPUs:

*   **Rome Partition (`tcn`):** dual AMD Rome 7H12 (128 cores total), Mellanox ConnectX-6 HDR100 (100 Gb/s) NICs.
*   **Genoa Partition (`tcn`):** dual AMD Genoa 9654 (192 cores total), Mellanox ConnectX-7 NDR400 NICs on a shared-cable topology (two nodes share one physical 400 Gb/s cable).
*   **A100 GPU Nodes (`gcn`):** dual Intel Xeon Platinum 8360Y, 4x NVIDIA A100-SXM4 (40GB) over NVLink (theoretical 100 GB/s D2D), dual ConnectX-6 NICs for 200 Gb/s inter-node.
*   **H100 GPU Nodes (`gcn`):** dual AMD EPYC 9334, 4x NVIDIA H100 (94GB HBM3), NDR ConnectX-7 (400 Gb/s).

---

## 2. The Container Network Cliff

We compared **native (bare-metal)** runs with two Apptainer (formerly Singularity) container models:
1.  **Hybrid:** the container brings its own application but binds the host's MPI and network stack at runtime.
2.  **Fully contained:** fully isolated. Compilers, OpenMPI, and UCX are built from source inside the image with **Spack**.

Point-to-point benchmarks (`Netgauge` and `OSU Micro-Benchmarks`) showed an unexpected cliff in the fully contained model:

*   **Native & Hybrid Genoa:** hit the expected peak (around 220 Gb/s).
*   **Fully contained Genoa:** dropped to **~10 Gb/s**, a 20x loss.
*   **Fully contained Rome:** a similar 10x drop to the same ~10 Gb/s.

### Why?
**UCX transport selection.** A fully isolated container cannot reach the host's kernel-space InfiniBand drivers (such as `libibverbs` or the Mellanox OpenFabrics drivers). When OpenMPI started UCX, UCX could not connect over InfiniBand verbs (`rc` or `ud` transports) and silently fell back to TCP over a slower Ethernet management interface.

---

## 3. Fixing Namespace and Permission Issues

HPC users have no root. Apptainer handles this with unprivileged user namespaces and `fakeroot` [36]. That security boundary breaks some of UCX's intra-node shared-memory tricks.

### `/proc` links
By default UCX shares memory descriptors between processes on a node through `/proc` links. Inside an unprivileged namespace that triggers security violations, so MPI init aborts or falls back to slower loopback.

The fix, without weakening host security, is to turn off procfs link sharing:
```bash
export UCX_POSIX_USE_PROC_LINK=n
```

### Vader BTL
OpenMPI's Byte Transfer Layer (BTL) also tries shared-memory optimizations that clash with containers. In our SLURM scripts:
```bash
# Disable Vader single-copy mechanisms that clash with container namespaces
export OMPI_MCA_btl_vader_single_copy_mechanism=none
```

---

## 4. Debugging UCX Wire-Up

Main lesson: **do not trust a containerized MPI run to use the fast network unless you verify it.** Our protocol:

### Step 1: List the transports UCX sees
Run `ucx_info -d` inside the container. If it shows only `tcp` or `self` and no Mellanox InfiniBand devices (such as `mlx5_0`), the host drivers are not mapped into the container.

For Apptainer we fixed this with the `--nv` flag (for GPUs) and, when needed, by binding the Mellanox device directories:
```bash
apptainer exec --nv final_hybrid.sif ...
```

### Step 2: Log connection decisions
UCX negotiates silently by default. Turn on `FI_INFO_LEVEL` and OpenMPI verbose MCA parameters:
```bash
mpirun -mca pml_ucx_verbose 1 \
       -mca btl_base_verbose 100 \
       -x FI_INFO_LEVEL=info \
       ./your_benchmark
```
The log shows whether UCX bound to `smCUDA` (CUDA-aware intra-node shared memory), direct InfiniBand verbs, or fell back to TCP.

### Step 3: Pin the interfaces
So UCX does not route over the wrong interface (Genoa nodes have several), we restricted the allowed ones:
```bash
# Rome Partition
export OMPI_MCA_btl_tcp_if_include=eno2np0,ib0

# Genoa Partition
export OMPI_MCA_btl_tcp_if_include=ens4f0np0,ib0
```

---

## 5. GPU Communication: CUDA-Aware MPI vs. NCCL

On the GPU partition, UCX powers **CUDA-Aware MPI** [39]: you pass GPU device pointers straight to MPI calls (`MPI_Send`, `MPI_Recv`) and skip the manual copy to the host (`Trivial Staging`).

We profiled power and bandwidth of CUDA-Aware MPI, NVIDIA NCCL, and CUDA IPC [40, 41] with a custom profiler built on direct **NVML** and **DCGM** API calls [15, 55].

### What we saw

*   **Peer-to-Peer (PP):** CUDA-Aware MPI (via UCX) and NCCL both kept up with native, reaching ~87 GiB/s on A100 (close to the 100 GB/s NVLink limit).
*   **AllReduce (AR):** here they diverged. NCCL used optimized ring/tree topologies over NVLink, while CUDA-Aware MPI failed to scale and dropped to **1 GiB/s** on A100 under heavy load.

The profiler's PCIe and NVLink counters showed why: under heavy `AllReduce` congestion, UCX's default transport arbitration failed to pick NVLink paths, so CUDA-Aware MPI sent traffic over the much narrower PCIe bus. I still have not pinned down exactly when that arbitration flips, but the counter traces made the fallback impossible to ignore.

The result: NVLink sat underused and CPU-side synchronization overhead went up.

---

## 6. Practical Takeaways

For communication-heavy containerized workloads on modern supercomputers:

1.  **Prefer the hybrid container model.** It is said to be harder to configure, but hybrid containers (binding the host's MPI/UCX at runtime) matched bare-metal performance. Fully contained images need constant Spack upkeep to match host driver versions and are prone to silent TCP fallbacks.
2.  **Always set the namespace flags.** Put these in your SLURM/PBS scripts when running Apptainer or Singularity:
    *   `UCX_POSIX_USE_PROC_LINK=n`
    *   `OMPI_MCA_btl_vader_single_copy_mechanism=none`
3.  **Verify with logs.** A finished job does not mean it ran on InfiniBand. Add a check with `ucx_info` and connection logging (`FI_INFO_LEVEL=info`) to your pipeline.

---

### References

* De Sensi et al., *Noise in the Clouds: Influence of Network Performance Variability on Application Scalability* (2022). [arXiv:2210.15315](https://arxiv.org/abs/2210.15315)
* De Sensi et al., *Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects* (IEEE SC24). [arXiv:2408.14090](https://arxiv.org/abs/2408.14090)
* Lang & Rünger, *High-Resolution Power Profiling of GPU Functions Using Low-Resolution Measurement* (Euro-Par 2013). [DOI](https://doi.org/10.1007/978-3-642-40047-6_80)
* Apptainer Contributors, [*Fakeroot Feature — Apptainer User Guide*](https://apptainer.org/docs/user/latest/fakeroot.html)
* Full list: [All references](/readinglist/).

---
