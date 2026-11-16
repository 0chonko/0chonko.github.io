---
title: "The Shared-Cable Bottleneck: Diagnosing Unexpected Interconnect Anomalies on AMD Genoa & ConnectX-7"
date: 2024-07-01
description: "An empirical investigation into why bare-metal Genoa nodes equipped with Mellanox NDR400 ConnectX-7 adapters suffered severe bandwidth drops on a shared-cable switch topology."
---

Cluster architects plan around vendor specs, but runtime numbers rarely match the theoretical limits.

During a network study on the Dutch National Supercomputer (*Snellius*), I hit a large bandwidth anomaly when comparing the newer **AMD Genoa** partition with the older **AMD Rome** partition. This post covers the debugging, the limits of shared-cable topologies, and how the Interquartile Range (IQR) helps separate real anomalies from multi-tenant noise.

## The Setup
Both CPU partitions sit on InfiniBand fat-tree topologies:
*   **Rome:** dual AMD Rome 7H12 (128 cores total), single-port ConnectX-6 HDR100 adapter (theoretical **100 Gb/s** inter-node).
*   **Genoa:** dual AMD Genoa 9654 (192 cores total), single-port NDR400 ConnectX-7 NIC (theoretical **400 Gb/s**).

We ran point-to-point unidirectional and bidirectional `Netgauge` and `OSU Micro-Benchmarks` on node pairs on the same switch and on different switches.

## The Anomaly: Over 85% Bandwidth Drop
On bare metal, Rome was predictable: a stable **98.1 Gb/s** ceiling within and across racks.

Genoa was not. Some runs peaked at **223 Gb/s**, but on the *same switch* it often dropped suddenly to **15 to 25 Gb/s**, more than 85% below what the specs suggest.

```text
Expected Genoa NDR400: ~200 Gb/s (Intra-Node) / 400 Gb/s (Inter-Node Cable Cap)
Observed Genoa Native:  223 Gb/s (Peak) -> 15-25 Gb/s (Degraded Swappings)
```

## Troubleshooting
We first suspected software: a mismatched or misconfigured MPI transport layer (UCX/OFI). To rule that out, we isolated hardware and topology step by step.

### 1. Shared-Cable Topology
Mapping the physical wiring of the Genoa partition showed a shared-cable design. To save ports on expensive switches, each pair of Genoa nodes shares one physical **NDR400 (400 Gb/s) cable**.

That is cheap and fine when traffic is staggered. But when both nodes on a split cable run bulk transfers at once, the link congests hard, which matches the **15 to 25 Gb/s** troughs we kept seeing.

![Shared Cable Architecture](/assets/blog/shared-cable-topology.png)

### 2. Same-Rack vs Cross-Rack Noise
To check whether this depended on switch placement, we ran 1-hour continuous benchmarks, recording latency and bandwidth noise and tracking the smallest 0.1% of samples to capture worst-case jitter.

We plotted the **Cumulative Distribution Function (CDF)** of latency and bandwidth and summarized each run with the **IQR**: the gap between the 25th (Q1) and 75th (Q3) percentiles, i.e. the middle half of the samples. Anything outside the usual Q1 − 1.5×IQR / Q3 + 1.5×IQR fence we treated as outliers from neighbor traffic or transient congestion.

Rome's bandwidth sat in a tight band around **~98 Gb/s** with a small IQR. Genoa's spread was much wider: one hour could include bursts near **223 Gb/s** and long troughs at **15 to 25 Gb/s**, with a clearly larger IQR even after removing the extreme tails.

Filtering outliers this way separated neighbor traffic on the shared fat-tree from Genoa's own instability. Genoa has the higher peak on paper, but much more **bandwidth noise** than the non-shared ConnectX-6 HDR100 links on Rome. Worth keeping in mind next to the UCX transport issues on the same machine: sometimes the problem is software, sometimes it is a cable you share with your neighbor.

## Key Takeaways
1.  **Watch out for shared cabling.** Split or shared links (like NDR shared links) make performance unpredictable. For communication-heavy jobs, check whether your nodes share physical switch links.
2.  **Use IQR, not averages.** For long noise measurements, plain averages and maximums mislead. The IQR shows typical behavior and resists outliers.
3.  **Compare same-switch and cross-switch.** This tells you whether a dip comes from local cabling or from a wider routing issue.

---

### References

* De Sensi et al., *Noise in the Clouds: Influence of Network Performance Variability on Application Scalability* (2022). [arXiv:2210.15315](https://arxiv.org/abs/2210.15315)
* De Sensi et al., *Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects* (IEEE SC24). [arXiv:2408.14090](https://arxiv.org/abs/2408.14090)
* Full list: [All references](/readinglist/).

---
