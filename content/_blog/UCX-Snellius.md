---
title: Experiences with UCX on the Dutch National Supercomputer
date: 2025-06-11
description: "Navigating the complexities of UCX transport selection, unprivileged container namespaces, and network performance debugging on the Snellius supercomputer."
---

Deploying high-performance computing (HPC) applications in containerized environments remains one of the most active areas of system optimization. While CPU and GPU virtualization overheads have largely been minimized, the real bottleneck often lies in the network. Specifically, how do containerized message-passing runtimes interact with proprietary network fabrics?

During the work on my Master's thesis at Vrije Universiteit Amsterdam, I conducted an empirical performance and energy characterization of the **Snellius** supercomputer (the Dutch National Supercomputer managed by SURF) [6, 15]. A significant portion of this research involved wrestling with **Unified Communication X (UCX)**, the middleware that bridges MPI runtimes (like OpenMPI) with high-speed interconnects (InfiniBand HDR/NDR and NVLink).

This post is what I wish I had read before the first benchmark looked fine but ran over TCP anyway: the debugging steps, the namespace flags, and the configs that actually got UCX working with Apptainer on Snellius CPU and GPU partitions.

---

## 1. The Hardware Infrastructure

Our testbed on Snellius spanned two generations of AMD EPYC CPU nodes and two generations of enterprise NVIDIA GPUs:

*   **Rome Partition (`tcn`):** Dual AMD Rome 7H12 processors (128 cores total), Mellanox ConnectX-6 HDR100 (100 Gb/s) NICs.
*   **Genoa Partition (`tcn`):** Dual AMD Genoa 9654 processors (192 cores total), Mellanox ConnectX-7 NDR400 NICs on a shared-cable topology (two nodes sharing one physical 400 Gb/s cable).
*   **A100 GPU Nodes (`gcn`):** Dual Intel Xeon Platinum 8360Y, 4x NVIDIA A100-SXM4 (40GB) interconnected via NVLink (theoretical 100 GB/s D2D), with dual ConnectX-6 NICs for a 200 Gb/s inter-node network.
*   **H100 GPU Nodes (`gcn`):** Dual AMD EPYC 9334, 4x NVIDIA H100 (94GB HBM3), NDR ConnectX-7 (400 Gb/s) interconnects.

---

## 2. The Containerized Network Bottleneck

In our experiments, we compared a **Native (bare-metal)** environment against two container deployment models built via Apptainer (formerly Singularity):
1.  **Hybrid Model:** The container utilizes its own user-space application but mounts and binds the host's MPI and network stack dynamically.
2.  **Fully Contained Model:** The container is completely isolated. The software stack (compilers, OpenMPI, UCX) is built entirely from source inside the image using the **Spack** package manager.

When running baseline point-to-point network benchmarks (using `Netgauge` and `OSU Micro-Benchmarks`), we observed a stark, unexpected performance cliff in the Fully Contained model.

*   **Native & Hybrid Genoa:** Reached their expected peaks (around 220 Gb/s).
*   **Fully Contained Genoa:** Dropped catastrophically to **~10 Gb/s**, a 20x performance loss.
*   **Fully Contained Rome:** Exhibited a similar 10x drop down to the same ~10 Gb/s threshold.

### Why did this happen?
The culprit was **UCX transport selection**. 

When a container runs in complete isolation, it lacks direct access to the host's proprietary kernel-space InfiniBand drivers (such as `libibverbs` or Mellanox OpenFabrics drivers). When OpenMPI initialized via UCX, UCX searched for available communication paths. Unable to establish a connection over InfiniBand verbs (`rc` or `ud` transports), UCX silently fell back to standard TCP over a slower Ethernet management interface. 

---

## 3. Resolving UCX Namespace & Permission Issues

In multi-tenant HPC environments, users do not have root privileges. Apptainer handles this by running containers using unprivileged user namespaces and unprivileged execution (`fakeroot`) [36]. However, this security boundary creates challenges for UCX's intra-node shared memory mechanisms.

### The `/proc` File System Collision
By default, UCX utilizes `/proc` filesystem links to share memory descriptors between processes on the same node. In an unprivileged container namespace, attempting to access these links triggers security violations, causing MPI initialization to abort or fall back to slower network loopbacks.

To bypass this without compromising the host's security model, we had to explicitly instruct UCX to disable procfs-based link sharing by setting:
```bash
export UCX_POSIX_USE_PROC_LINK=n
```

### Shared Memory and Vader BTL
Additionally, within OpenMPI, we had to fine-tune the Byte Transfer Layer (BTL) to prevent it from attempting container-unfriendly shared memory optimizations. In our SLURM batch scripts, we injected:
```bash
# Disable Vader single-copy mechanisms that clash with container namespaces
export OMPI_MCA_btl_vader_single_copy_mechanism=none
```

---

## 4. Deep-Dive: Debugging UCX Wire-Up

One of the key lessons from this work is that **you cannot trust a containerized MPI execution to run over the fast network path unless you explicitly verify it.** 

When performance issues arose, we relied on a strict debugging protocol:

### Step 1: Querying UCX Recognized Transports
We ran `ucx_info -d` inside the container environment to verify which hardware devices and transports UCX actually recognized. 
If `ucx_info -d` returned only `tcp` or `self` devices and did not list Mellanox InfiniBand devices (such as `mlx5_0`), it meant the host drivers were not mapped into the container correctly. 

For Apptainer, we corrected this by ensuring the `--nv` flag (for GPUs) was set, and when necessary, explicitly bound the Mellanox device directories:
```bash
apptainer exec --nv final_hybrid.sif ...
```

### Step 2: Logging UCX Connection Decisions
By default, UCX negotiates connections silently. To force it to reveal its connection mapping, we turned on `FI_INFO_LEVEL` and OpenMPI verbose MCA parameters:
```bash
mpirun -mca pml_ucx_verbose 1 \
       -mca btl_base_verbose 100 \
       -x FI_INFO_LEVEL=info \
       ./your_benchmark
```
This verbose output explicitly logs whether UCX successfully binds to `smCUDA` (for CUDA-aware intra-node shared memory), direct InfiniBand verbs, or if it falls back to TCP.

### Step 3: Interface Inclusion
To prevent UCX from accidentally routing traffic over the wrong network interface (especially on Genoa nodes where multiple interfaces exist), we strictly bounded the allowed interfaces:
```bash
# Rome Partition
export OMPI_MCA_btl_tcp_if_include=eno2np0,ib0

# Genoa Partition
export OMPI_MCA_btl_tcp_if_include=ens4f0np0,ib0
```

---

## 5. GPU Communications: CUDA-Aware MPI vs. NCCL

On the GPU partition, UCX acts as the critical engine behind **CUDA-Aware MPI** [39]. It allows developers to pass GPU device pointers directly to MPI calls (such as `MPI_Send` and `MPI_Recv`), bypassing the traditional, expensive manual copy to the host CPU (`Trivial Staging`).

We profiled the power draw and bandwidth of different communication libraries (CUDA-Aware MPI, NVIDIA's NCCL, and CUDA IPC) [40, 41] using a custom-built profiler leveraging direct **NVML** and **DCGM** API calls [15, 55].

### Microarchitectural Insights

*   **Peer-to-Peer (PP):** Both CUDA-Aware MPI (via UCX) and NCCL kept up with the native environment well, achieving ~87 GiB/s on A100 (close to the 100 GB/s NVLink physical limit).
*   **AllReduce (AR) Congestion:** In complex, multi-GPU collective communications like `AllReduce`, we noticed a distinct divergence. While NCCL successfully utilized highly optimized ring/tree topologies over NVLink, CUDA-Aware MPI struggled to scale, dropping to **1 GiB/s** on A100 under heavy workloads.

Our profiler's PCIe and NVLINK hardware counters revealed why: under high congestion during `AllReduce` collectives, UCX's default transport arbitration logic struggled to resolve NVLink paths, causing CUDA-Aware MPI to route traffic over the much narrower PCIe bus instead. I still haven't pinned down exactly when that arbitration flips, but the counter traces made the fallback impossible to ignore.

This resulted in significant NVLink underutilization and elevated CPU-side synchronization overhead.

---

## 6. Practical Takeaways for HPC Performance Engineers

For researchers looking to run communication-dense containerized workloads on modern supercomputers, my experience suggests the following guidelines:

1.  **Prefer the Hybrid Container Model:** Although marketed as more complex to configure, hybrid containers (which dynamically bind the host's MPI/UCX libraries) consistently matched native bare-metal performance. Fully contained models require tedious maintenance of Spack configurations to match host driver versions, and are highly prone to silent TCP fallbacks.
2.  **Enforce Safe Namespace Environment Flags:** Always include namespace workarounds in your SLURM/PBS submit scripts when running Apptainer or Singularity:
    *   `UCX_POSIX_USE_PROC_LINK=n`
    *   `OMPI_MCA_btl_vader_single_copy_mechanism=none`
3.  **Validate via Log Tracing:** Never assume your container is on InfiniBand just because the job completes. Build a verification step into your pipeline using `ucx_info` and connection-level logging (`FI_INFO_LEVEL=info`).

---

### References

* De Sensi et al., *Noise in the Clouds: Influence of Network Performance Variability on Application Scalability* (2022). [arXiv:2210.15315](https://arxiv.org/abs/2210.15315)
* De Sensi et al., *Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects* (IEEE SC24). [arXiv:2408.14090](https://arxiv.org/abs/2408.14090)
* Lang & Rünger, *High-Resolution Power Profiling of GPU Functions Using Low-Resolution Measurement* (Euro-Par 2013). [DOI](https://doi.org/10.1007/978-3-642-40047-6_80)
* Apptainer Contributors, [*Fakeroot Feature — Apptainer User Guide*](https://apptainer.org/docs/user/latest/fakeroot.html)
* Full list: [All references](/readinglist/).

---


