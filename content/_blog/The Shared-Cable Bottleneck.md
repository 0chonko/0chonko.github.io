---
title: "The Shared-Cable Bottleneck: Diagnosing Unexpected Interconnect Anomalies on AMD Genoa & ConnectX-7"
date: 2024-07-01
description: "An empirical investigation into why bare-metal Genoa nodes equipped with Mellanox NDR400 ConnectX-7 adapters suffered severe bandwidth drops on a shared-cable switch topology."
---

When configuring multi-million-dollar high-performance computing (HPC) clusters, system architects rely heavily on vendor specifications. However, as any performance engineer knows, actual runtime metrics rarely align cleanly with theoretical limits. 

During a network characterization study on the Dutch National Supercomputer (*Snellius*), I encountered a major bandwidth anomaly while evaluating the newer **AMD Genoa** partition against the older **AMD Rome** partition. This post documents the debugging process, the hardware limitations of shared-cable topologies, and how statistical tools like the Interquartile Range (IQR) help isolate real network anomalies from multi-tenant noise.

## The Benchmark Arena
Our experiments targeted two distinct CPU partitions connected via InfiniBand fat-tree topologies:
*   **Rome Partition:** Dual AMD Rome 7H12 processors (128 cores total), single-port ConnectX-6 HDR100 adapter (capped at a theoretical **100 Gb/s** inter-node).
*   **Genoa Partition:** Dual AMD Genoa 9654 processors (192 cores total), single-port NDR400 ConnectX-7 network interface card (theoretically capable of **400 Gb/s**).

To capture clean network metrics, we ran point-to-point bidirectional and unidirectional `Netgauge` and `OSU Micro-Benchmarks` on pairs of nodes distributed across the same switch or different switches.

## The Anomaly: Over 85% Drop in Bandwidth
Under native (bare-metal) execution, the Rome nodes behaved predictably, hitting a stable ceiling of **98.1 Gb/s** on both intra-rack and inter-rack tests. 

However, the Genoa partition threw a massive curveball. While we recorded impressive peak bursts of **223 Gb/s** on some runs, the native Genoa setup on the *same switch* frequently hit sudden, severe degradations, dropping to **15 to 25 Gb/s**, over an 85% reduction from what the specs suggest.

```text
Expected Genoa NDR400: ~200 Gb/s (Intra-Node) / 400 Gb/s (Inter-Node Cable Cap)
Observed Genoa Native:  223 Gb/s (Peak) -> 15-25 Gb/s (Degraded Swappings)
```

## Troubleshooting the Bottleneck
Initially, we suspected a software mismatch or a misconfigured MPI transport layer (UCX/OFI). To rule this out, we systematically isolated the hardware and topology:

### 1. Shared Cable Topology Analysis
By digging into the physical wire-up of the Genoa partition on Snellius, we mapped out a shared-cable architecture. To maximize port density on expensive high-speed switches, each pair of Genoa nodes shares a single physical **NDR400 (400 Gb/s) network cable**. 

While this configuration is cost-effective and works well when workloads are staggered, it creates a physical bottleneck during concurrent communication bursts. If Node A and Node B share a split-cable interface and both try a bulk transfer at once, the link congests hard, dragging throughput down to the **15 to 25 Gb/s** range we kept seeing.

![Shared Cable Architecture](/assets/blog/shared-cable-topology.png)

### 2. Differentiating Intra-Rack and Inter-Rack Noise
To confirm whether this issue was localized to specific switch placements, we ran continuous 1-hour benchmarks. We recorded Latency and Bandwidth noise, tracking the smallest 0.1% of collected samples to capture the worst-case jitter.

To distinguish systemic architectural anomalies from standard multi-tenant network noise, we plotted the **Cumulative Distribution Function (CDF)** of latency and bandwidth, then summarized each run with the **Interquartile Range (IQR)**: the gap between the 25th percentile (Q1) and the 75th percentile (Q3). That window holds the middle half of samples; anything outside the usual Q1 − 1.5×IQR / Q3 + 1.5×IQR fence we treated as outliers from neighbor traffic or transient congestion.

On Rome, bandwidth samples clustered in a tight band around **~98 Gb/s** with a small IQR. Genoa's distribution was much wider: the same hour-long window could include bursts near **223 Gb/s** and sustained troughs in the **15 to 25 Gb/s** range, with a noticeably larger IQR once we stripped the extreme tails.

That outlier filtering let us separate neighbor traffic on the shared fat-tree from the Genoa partition's own instability. The data confirmed that Genoa yields higher peak throughput on paper, but also significantly more **bandwidth noise** and less stability than the mature, non-shared ConnectX-6 HDR100 links on Rome. That noise profile is worth keeping in mind next to the UCX transport issues on the same machine: sometimes the problem is software, sometimes it is a cable you share with your neighbor.

## Key Takeaways
1.  **Beware of High-Density Cabling:** Split-cable or shared-cable topologies (like NDR shared links) can introduce massive performance unpredictability. If your HPC workload is communication-dense, verify whether your target nodes share physical switch links.
2.  **Filter with IQR:** When evaluating network noise over extended periods, avoid relying on simple averages or absolute maximums alone. The Interquartile Range gives an outlier-resistant picture of typical behavior.
3.  **Validate Across Switches:** Always contrast same-switch performance against different-switch configurations to isolate whether a performance dip is a localized cabling bottleneck or a broader routing path issue.

---

### References

* De Sensi et al., *Noise in the Clouds: Influence of Network Performance Variability on Application Scalability* (2022). [arXiv:2210.15315](https://arxiv.org/abs/2210.15315)
* De Sensi et al., *Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects* (IEEE SC24). [arXiv:2408.14090](https://arxiv.org/abs/2408.14090)
* Full list: [All references](/readinglist/).

---
