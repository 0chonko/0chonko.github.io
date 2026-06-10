---
title: "Demystifying GASPI: Notes from a 2-Day HLRS Tutorial " 
date: 2026-06-19
description: "What I took away from HLRS's GASPI course on PGAS programming, one-sided RDMA, and why global barriers might be the wrong abstraction at scale. [Certificate](/assets/blog/demystifying-gaspi-hlrs-tutorial/personal-certificate.pdf)"
---

Most of my hands-on HPC work so far has lived inside the **MPI** world. OpenMPI, UCX transport selection, verifying that a containerized job actually hits InfiniBand instead of silently falling back to TCP. That stack is familiar, and honestly a bit comfortable in the way familiar things are until they stop scaling.

I spent two days in an online tutorial at **HLRS** (the High Performance Computing Center Stuttgart) on **GASPI** (Global Address Space Programming Interface), and it reframed a question I keep running into: what if the bottleneck is not the network hardware itself, but the communication *model* we wrap around it?

GASPI is a standardized PGAS API built around one-sided RDMA, asynchronous queues, and fine-grained notifications instead of cooperative two-sided sends and global barriers. This post is my condensed notes from the course, plus where I think it connects to the MPI-centric debugging work I have been doing elsewhere.

---

## Why Bother with Another Communication API?

At scale, communication stops being a side detail. It costs latency, memory bandwidth, energy, and operator money. The tutorial framed this bluntly: traditional message-passing patterns often serialize compute and communication in ways the hardware never asked for.

GASPI's pitch is not "replace MPI everywhere tomorrow." It is closer to giving you predictable, low-level building blocks so you can structure parallelism around **data dependencies** rather than phase boundaries. Five design guidelines kept coming back across both days:

1. **Overlap communication and computation.** Post transfers asynchronously and keep the CPU doing useful work while the NIC moves data.
2. **Zero copy.** Application buffers go straight to remote memory where possible. No staging through extra bounce buffers on the CPU.
3. **No automatic synchronization.** Global barriers amplify load imbalance and OS jitter. Synchronize only along real dependency edges, and keep those groups small.
4. **Exploit the communication hierarchy.** Core, NUMA node, socket, node, network: traffic that can stay local should stay local.
5. **One-sided communication.** The initiator specifies source, target, and size. The remote CPU does not need to enter a matching receive.

That last point clicked immediately given my UCX debugging on Snellius. I spent a lot of time proving that MPI+UCX actually wired up to InfiniBand. GASPI assumes you want that kind of direct hardware path by default, not as something you verify after the fact.

---

## PGAS and Segments: Memory You Can Name Globally

Under the hood, GASPI is a **Partitioned Global Address Space (PGAS)** model. You still know memory is physically distributed across nodes, but you can reference remote partitions through a unified logical address space.

```text
               GLOBAL ADDRESS SPACE (PGAS)
  +---------------------------------------------------+
  |  Node 1 RAM  |  Node 2 RAM  | ...  |  Node N RAM  |
  +--------------+--------------+------+--------------+
     (Local)        (Remote)               (Remote)
```

The practical abstraction is the **segment**: a contiguous block of virtual memory registered with the NIC.

* **`gaspi_segment_alloc`** allocates and pins memory so it stays resident for **RDMA**.
* **`gaspi_segment_register`** makes the segment visible for local pointer access and remote one-sided operations.

Segments are the foundation. Reads, writes, and notifications all operate on them.

---

## The API Pillars That Actually Matter

The surface area is intentionally small. Three ideas do most of the work.

### One-Sided Transfers

Non-blocking puts and gets between local and remote segments:

* **`gaspi_write`**: post a put from local to remote memory.
* **`gaspi_read`**: post a get from remote into local memory.

Both return immediately. Completion is tracked through queues and waits, not blocking call semantics.

### Queues

Every read/write goes through a local **communication queue**. Queues are finite; a full queue returns `GASPI_QUEUE_FULL`.

**`gaspi_wait`** on a queue flushes it: locally complete means incoming read data is visible and source buffers from completed writes can be reused. This is the point where you trade fire-and-forget posting for actual progress guarantees.

### Notifications

One-sided transfers do not wake the remote CPU. So how does the other side know data landed?

* **`gaspi_notify`**: atomically signal a notification ID on a remote segment.
* **`gaspi_notify_waitsome`**: block (or time out) until selected notification IDs become non-zero.
* **`gaspi_notify_reset`**: reset an ID and return its previous value.

The tutorial emphasized **`gaspi_write_notify`** and **`gaspi_write_list_notify`**, hardware-oriented fused operations that combine a transfer and a signal in one shot. Fewer round trips, less synchronization surface area.

| Mechanism | Role |
| :--- | :--- |
| `gaspi_write` / `gaspi_read` | Async data movement between segments |
| Queue + `gaspi_wait` | Local completion and buffer reuse |
| `gaspi_notify*` | Remote readiness without a matching receive |

---

## From Bulk Synchronous to Dataflow

The section that stuck with me most was not a single API call. It was the execution model shift.

A classic stencil or 1D ring exchange in MPI often looks like this:

```
[ Phase 1: Communicate ] -> [ Global Barrier ] -> [ Phase 2: Compute ] -> [ Global Barrier ]
```

Simple to reason about. Wasteful in practice. CPUs idle while the network works, then the network idles while CPUs work. Global barriers also collect every node's jitter into one synchronization point, which is rough on large, heterogeneous partitions.

The GASPI pattern splits the domain into **inner** cells (no remote deps) and **boundary** cells (need neighbor data):

1. Post boundary transfers with `gaspi_write_notify`.
2. Compute inner cells while the NIC handles boundaries in the background.
3. Wait on specific notifications via `gaspi_notify_waitsome`, not a world barrier.
4. Update boundaries once data is confirmed.

Local stage counters per subdomain replace process-wide phases. Tasks run when their dependencies are satisfied, not when the slowest rank catches up. The tutorial claimed near-optimal overlap when the split is done well. I have not benchmarked that myself yet, but the logic parallels what I saw when profiling congestion on GPU collectives: the expensive part is often waiting on the wrong synchronization primitive, not raw link speed.

---

## Mixed Mode with MPI

Full rewrites are unrealistic for most production codes. GASPI supports **incremental porting** alongside MPI:

* Ranks align automatically with the existing MPI layout at runtime.
* **`gaspi_segment_use`** can register buffers MPI already allocated, avoiding duplicate allocations and extra copies.

```
MPI_Init(&argc, &argv);
gaspi_proc_init(GASPI_BLOCK);

// heterogeneous MPI + GASPI work here

gaspi_proc_term(GASPI_BLOCK);
MPI_Finalize();
```

This feels like the realistic on-ramp. Keep MPI for coarse structure and I/O, drop GASPI into the hot loop where latency and overlap matter. Given how much effort went into hybrid container setups on Snellius, I am curious whether a mixed-mode MPI+GASPI path would interact cleanly with bound host libraries or whether segment registration would hit the same namespace/driver visibility issues I saw with fully contained images.

---

## What This Connects To (and What I Have Not Tried)

The HLRS material sits adjacent to several threads from my own work without replacing any of them:

* **UCX and MPI on Snellius** taught me that the fast path is never automatic. GASPI pushes responsibility earlier: you design for RDMA and explicit dependencies up front instead of discovering a TCP fallback after a benchmark looks wrong.
* **Shared-cable topology effects** showed how physical hierarchy shows up in bandwidth noise. GASPI's "exploit the communication hierarchy" guideline is the software-side mirror of that same constraint.
* **GPU power profiling** highlighted observer effects in microsecond-scale measurements. GASPI's async model is the communication analogue: do not block the critical path waiting for things that could overlap.

What I do not know yet, and what I would want to test on real hardware: does the notification-driven stencil pattern actually beat a well-tuned MPI+UCX implementation on the same partition? And for mixed mode, where does segment registration break when containers isolate user-space drivers? Those feel like the natural next experiments, not more slide decks.

---

## Practical Takeaways

1. **Treat barriers as a design smell.** If you reach for `MPI_Barrier` to separate communicate/compute phases, ask whether a dependency-local notification would shrink the idle window.
2. **Think in segments, not messages.** Pinned, registered memory is the unit of fast transfer. Buffer lifetime and queue completion become part of your mental model, like rank topology already is in MPI.
3. **Overlap before you optimize bandwidth.** The tutorial's core win is structural: inner/boundary splitting plus async writes buys parallelism that a phased BSP loop leaves on the table.
4. **Port incrementally.** `gaspi_segment_use` and rank-matched MPI startup lower the migration cost. You do not need a greenfield rewrite to probe whether one kernel benefits.
5. **Verify on your machine.** Same lesson as UCX: APIs promise hardware paths; your job is to confirm the path you think you are on is the one the runtime actually took.

---

### References

* Fraunhofer ITWM / HLRS, *GASPI Tutorial* (2-day online course, 2026). [Certificate](/assets/blog/demystifying-gaspi-hlrs-tutorial/personal-certificate.pdf).
* GASPI Consortium, [GASPI Standard and Documentation](https://www.gaspi.de/).
* Full list: [All references](/readinglist/).

