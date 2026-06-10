---
title: "Green GPU Communication: Profiling the Watts/GB of NVLink vs. PCIe Across A100 and H100 Architectures"
date: 2024-09-15
description: "A deep dive into the energy footprint of GPU interconnects, exploring the thermodynamic cost of inter-device transfers."
---

As large language models scale to hundreds of billions of parameters, training clusters get big enough that communication stops being a latency problem alone. It shows up on the power bill too.

I used the async power profiler from my earlier GPU work on *Snellius* to measure Watts and memory utilization across **NVIDIA A100 (Ampere)** and **H100 (Hopper)** nodes, comparing **NVLink**, **PCIe**, **CUDA-Aware MPI**, and **NCCL**.

---

## The Testing Topology
We analyzed two modern GPU node configurations:
*   **NVIDIA A100 SXM4 Node:** Equipped with 4x A100 GPUs (40GB HBM2). Inter-device communication is facilitated by **NVLink Gen3** (4 links per GPU, providing a theoretical 100 GB/s bidirectional bandwidth).
*   **NVIDIA H100 Node:** Equipped with 4x H100 GPUs (94GB HBM3). Inter-device links run on **NVLink Gen4** (providing up to 160 GB/s per device for peer-to-peer transfers).

We tested four primary software pathways to move data:
1.  **Trivial Staging:** Manual copy from GPU to Host CPU, transferred via CPU MPI, then copied back down to the target GPU.
2.  **CUDA-Aware MPI:** Passing GPU device pointers directly to MPI, allowing the runtime to coordinate direct transfers via UCX.
3.  **NCCL (NVIDIA Collective Communications Library):** NVIDIA's highly proprietary collective library.
4.  **CUDA IPC:** Direct memory access between GPUs on the same node via NVLink.

---

## Finding 1: The NVLink Energy Efficiency Dividend
One of the most immediate takeaways from our energy measurements was the confirmation of the NVLink energy efficiency dividend. 

Because **NVLink** gives higher bandwidth and lower latency than **PCIe**, it finishes transfers faster. The GPU gets back to a lower idle power state sooner, what architects call **"race-to-idle."**

Our profiling showed that NVLink paths via **NCCL** or **CUDA IPC** can cut total energy by **up to 60%** versus PCIe-bound options like Trivial Staging, for the same payload sizes.

```text
Total Energy Consumption (Joules):
[PCIe Trivial Staging]   ======================================== (High Energy, Slow)
[NVLink NCCL]            ================ (Low Energy, High Bandwidth Race-To-Idle)
```

---

## Finding 2: Power Footprint of Communication Patterns
We profiled three distinct communication patterns: **Peer-To-Peer (PP)**, **All-To-All (A2A)**, and **All-Reduce (AR)** across varying payload sizes (up to 0.5 GB).

### 1. Peer-To-Peer (PP) Power Profile
For simple P2P transfers, NCCL and CUDA IPC demonstrated the lowest power draw. Interestingly, the lowest power draw of all was achieved by **CUDA IPC**, as it bypasses the CPU and MPI runtimes entirely, executing direct `cudaMemcpyPeer` operations over NVLink.

### 2. The Collective Burden: All-To-All & All-Reduce
As the communication patterns became more complex, the differences in power draw grew pronounced. 

```
Power Draw on A100 during All-Reduce (Peak Watts):
NCCL:           ~142.95 W 
CUDA-Aware MPI: ~125.00 W (But with significantly lower bandwidth and longer execution time)
CUDA IPC:       ~195.00 W (High immediate draw, but highly optimized)
```

During an `All-Reduce` operation on the A100, **NCCL** draw peaked at **142.95W**. In contrast, **CUDA IPC** peaked at **195W**. While CUDA IPC draws more instantaneous power, its memory allocation and link utilization are highly intensive, meaning it finishes the transfer much faster, resulting in lower net energy usage (Joules) over time.

---

## Finding 3: Memory Component and SM Utilization
Using DCGM hardware counters, we tracked **Streaming Multiprocessor (SM) Activity** (`DCGM_FI_PROF_SM_ACTIVE`) and **SM Occupancy** during these transfers. 

An interesting architectural division emerged:
*   **NCCL's Collective Domination:** For `All-Reduce`, NCCL peaked at approximately 16% SM activation, indicating highly structured, topology-aware algorithms that keep the SMs active just enough to orchestrate the transfers without wasting compute cycles.
*   **CUDA-Aware MPI PCIe Fallback:** During complex collectives, CUDA-Aware MPI fell back to PCIe instead of NVLink when queues congested. SM activity dropped toward 0% while the GPU waited on slow PCIe traffic. Same pattern I saw in the UCX bandwidth work, just read through a power meter this time.

---

## A100 vs. H100: Generational Energy Metrics
Comparing the two nodes, the **H100** demonstrated significantly greater power and runtime stability. 

While the A100 showed variable power drops under containerized environments, the H100 kept its performance and power metrics tightly aligned between native and containerized runs. This suggests that the Hopper architecture's onboard memory controllers and advanced NVSwitch fabrics are much more resilient to virtualization overhead.

## Key Performance Engineering Takeaways
1.  **Optimize for Race-To-Idle:** If you are trying to cut energy costs in large GPU clusters, prioritize peak throughput via **NCCL** or **CUDA IPC** to allow the GPUs to return to idle states quickly.
2.  **Monitor Your Interconnects:** Use DCGM counters (`DCGM_FI_PROF_PCIE_TX_BYTES` vs. `DCGM_FI_PROF_NVLINK_TX_BYTES`) to verify that your MPI runtime is not silently routing collective traffic over the slow PCIe bus during congestion.
3.  **Choose Hardware Consistently:** If cost estimation and budget predictability are paramount, the **H100** provides a much more stable and predictable power-to-performance profile under virtualized containers than the A100.

---

### References

* Lang & Rünger, *High-Resolution Power Profiling of GPU Functions Using Low-Resolution Measurement* (Euro-Par 2013). [DOI](https://doi.org/10.1007/978-3-642-40047-6_80)
* De Sensi et al., *Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects* (IEEE SC24). [arXiv:2408.14090](https://arxiv.org/abs/2408.14090)
* Full list: [All references](/readinglist/).