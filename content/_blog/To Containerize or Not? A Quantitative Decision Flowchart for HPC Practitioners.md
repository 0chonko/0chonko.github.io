---
title: "To Containerize or Not? A Quantitative Decision Flowchart for HPC Practitioners"
date: 2024-04-01
description: "A pragmatic guide to evaluating the network performance and deployment trade-offs of Apptainer, Spack, and Hybrid container models in high-performance computing."
---

In enterprise software, containerization is mostly a solved problem. Docker is the default, and the overhead is usually negligible.

In HPC it is a different story. Unprivileged namespaces, proprietary fabrics (InfiniBand, Cray Slingshot), and site-specific drivers mean you can tank network performance by **10x to 20x** if you containerize the wrong way.

From benchmarks on *Snellius*, this post walks through the trade-offs between container models and a decision flowchart for when (and how) to containerize.

---

## The Contenders: Container Models in HPC

When deploying containerized workloads via **Apptainer** (formerly Singularity), we generally choose between three models:

```text
                  HPC CONTAINER MODELS
                           |
       +-------------------+-------------------+
       |                   |                   |
 [Hybrid Model]       [Bind Model]     [Fully Contained]
  Uses host's          Mounts host      Isolates everything,
  network/MPI stack    directories      Spack-compiled
```

### 1. The Hybrid Model
The container packages your user-space application code, but at runtime, you dynamically mount the host supercomputer's MPI and network drivers.
*   **Pros:** Outstanding performance. Our benchmarks showed the Hybrid model matching native bare-metal speeds (peaking at **220 Gb/s** on Genoa ConnectX-7).
*   **Cons:** Poor portability. The container is tightly coupled to the host's software stack; running it on a different cluster with a different MPI version will often result in dynamic linking errors.

### 2. The Fully Contained Model
The container is completely isolated. All dependencies, including compilers, MPI runtimes, and UCX libraries, are built from source inside the image (typically with **Spack**).
*   **Pros:** Perfect portability and reproducibility. 
*   **Cons:** Severe performance drops if misconfigured. Without complex network wire-up, the containerized MPI cannot talk to the host's InfiniBand drivers, defaulting to slow TCP over Ethernet (dropping bandwidth from **220 Gb/s to a mere 10 Gb/s**).

---

## Spack: Composing the Fully Contained Stack
To build a functional Fully Contained container, we utilized Spack to generate optimized binaries. Below is an example of a Spack configuration (`spack.yaml`) used to build a containerized OpenMPI stack with CUDA support, targeting an NVIDIA A100 GPU:

```
spack:
  specs:
  - gmake@4.3
  - openmpi@4.1.5 fabrics=ucx,ofi +pmi +legacylaunchers +orterunprefix +cuda cuda_arch=80
  - libfabric fabrics=sockets,tcp,udp,psm2,verbs,mlx
  - ucx
  - slurm
  - cuda@12.1.0
  - nccl +cuda cuda_arch=80
  - pmix
  concretizer:
    unify: true
  config:
    install_tree: /opt/software
  view: /opt/view
```

Even with an optimized Spack specification, if the container runtime does not mount the host's NVIDIA device drivers (using Apptainer's `--nv` flag), the container's CUDA library will clash with the host's driver, causing runtime initialization failures.

---

## The Decision Flowchart: A Pragmatic Guide
To simplify this decision-making process, I synthesized our experimental findings into a structured flowchart:

```
              [ Start: Evaluate HPC Application ]
                              |
         Is the app a pure C/C++/Fortran program
         with minimal external dependencies?
                    /                   \
                 (Yes)                  (No)
                  /                       \
         [ Don't use a container ]    Is the program going to be
         (Run natively on host)       distributed alongside other
                                      containerized applications?
                                            /           \
                                         (Yes)          (No)
                                          /               \
                          [ Basic Contained ]      Does it scale to more
                                                   than one compute node?
                                                         /         \
                                                      (Yes)        (No)
                                                        /             \
                                        Does a ~10x hit to     [Any viable shared]
                                        latency/bandwidth       [memory parallelism]
                                        affect the application?        |
                                              /         \         [Don't use MPI]
                                           (Yes)        (No)
                                            /             \
                       Can you replicate the host's   [ Basic Contained ]
                       MPI stack using public repos?
                             /              \
                          (Yes)             (No)
                            /                 \
                  [ Optimized Contained ]   [ Hybrid Model ]
```

### Explaining the Decision Logic:

1.  **The Minimal Dependency Rule:** If your application is a straightforward C/Fortran code with basic library requirements, skip containerization. Running natively yields the lowest possible noise and avoids the complexities of unprivileged namespaces.
2.  **The Single-Node Exception:** If your workload does not scale past a single node (meaning it relies on OpenMP or shared memory instead of MPI), a **Basic Contained** model is perfectly adequate. There is no InfiniBand wire-up to worry about.
3.  **The Latency-Sensitive Check:** If your multi-node application is highly communication-bound, you must choose between **Optimized Contained** (using Spack to carefully compile UCX/InfiniBand drivers that match the host kernel) or the **Hybrid Model** (mounting the host's MPI). 

---

## Summary of Findings
*   **The Hybrid Model wins on performance:** For communication-dense workloads, the Hybrid model represents the best balance between ease of installation and close-to-native performance.
*   **The Fully Contained Model wins on portability:** However, it requires significant maintenance to ensure that UCX, Spack, and the host's driver versions align. 

By using this decision tree, you can skip a lot of the trial-and-error I went through on Snellius and pick a deployment model that matches how your application actually scales.

---

### References

* De Sensi et al., *Noise in the Clouds: Influence of Network Performance Variability on Application Scalability* (2022). [arXiv:2210.15315](https://arxiv.org/abs/2210.15315)
* De Sensi et al., *Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects* (IEEE SC24). [arXiv:2408.14090](https://arxiv.org/abs/2408.14090)
* Apptainer Contributors, [*Fakeroot Feature — Apptainer User Guide*](https://apptainer.org/docs/user/latest/fakeroot.html)
* Full list: [All references](/readinglist/).

---
