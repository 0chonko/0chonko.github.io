---
title: "A Vision for Malleable Machine Intuition"
date: 2026-09-22
description: "Intuition as craft already in the weights: sparse routing, write-then-consolidate memory, and a generation loop that grows competence inside the network."
---

After reading around Mixture-of-Experts, I kept landing on the same feeling: a **modular, composable, sparse** system is where efficient intelligence seems to go. Brains work that way. Not everything fires all the time. Specialists wake when they are needed.

What I mean by **intuition** is the craft already in the weights, the feel a person builds by practicing, not a fact fetched again at inference time. The picture I keep sketching is closer to Neo learning kung fu than Neo remembering the rules of kung fu. The interesting move would be to plug a learned feel into the model on demand, at the weights or layer level, or to change the architecture so memory itself is built differently. The questions that keep coming back are **routing**, **memory**, and **the loop**.

## Routing

Sparsity looks like an energy budget: a cortex cannot afford to have most of its neurons active at once [[1](https://www.cns.nyu.edu/~pl/pubs/Lennie03a.pdf)]. The sparse design I keep coming back to is DeepSeekMoE: shared experts that stay on, and fine-grained specialists that wake for the token [[2](https://aclanthology.org/2024.acl-long.70.pdf)].

We are still far from anything that looks like a clean thalamus or prefrontal cortex. The router looks like too much of a bottleneck to stand in for that. Continual learning barely hangs onto the network itself. Most deployed systems stay frozen and pull knowledge in from the outside, because serving a fixed model at scale is the reliable path.

## Memory

**Intuition**, in this reading, looks like a joint product of attention and memory. Attention is the short accurate window: what gets looked at now, prompt plus KV cache, give or take. Memory is the rest, and it is not one thing.

Working context seems to decay into **context rot** if you only keep stuffing it. **Episodic** memory would hold the instance (this session, this machine, that failure), grounded in time and place instead of smoothed into a generic fact. **Parametric / semantic** memory is the compressed craft already in the weights. Geva, Schuster, Berant, and Levy read the feed-forward layer as keys and values [[3](https://aclanthology.org/2021.emnlp-main.446.pdf)]. External retrieval sits outside that loop.

The status quo leans on that external path and hopes attention can re-read enough. That avoids writing into the network, and it also never turns practice into a feel that does not need to be fetched again. RAG-style knowledge is the usual way to give a model a specific case, mostly because frozen weights at scale are the easy, reliable serve.

The consolidation path that keeps coming up: write the episode while it is still sharp, then distill many narrow writes into fewer semantic ones. McClelland, McNaughton, and O’Reilly is the paper I have for that, day write, night distill [[4](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)]. Wake the right specialist when the routing, or the attention pattern, says so. On the test-time side, Titans keeps attention as the short window and writes a separate memory when surprise says so [[5](https://arxiv.org/pdf/2501.00663)]. End-to-end test-time training treats the long context as continual learning and trains it into the weights instead of re-reading it on every pass [[6](https://arxiv.org/pdf/2512.23675)].

## The loop

What keeps bugging me is how current models learn. Continual learning still looks like disconnected pieces: test-time training, inference-time weight updates, some control over how the network changes. Harnesses, agents, and tools keep suggesting that models do better when they get to run their own prediction loop, less steered from the outside.

Put together, the loop is still **autoregressive**: pick the next token and see what happens, then hang **auxiliary systems** around the model so the agent stays inside a bounding box. That scaffolds a result. It does not look like the competence growing inside the network.

A closer step is a diffusion language model like Mercury [[7](https://arxiv.org/pdf/2506.17298)]. The sequence starts noisy, and the model refines it coarse to fine, changing many positions in the same pass instead of reviewing tokens one by one. The dial is how many of those passes you run: fewer passes, a rougher draft, more passes, a finer one. In the usual masked setup, positions that are already committed tend to stay put, and what still moves is the rest of the field. So the cost flips away from "one token, then the next" and toward how much refinement you spend on the whole layout. Write the shape, then clean it, rather than walk the tokens in order.

## Fast and slow

The first thing in a long time that made me sit up was hearing about [**JEV**](https://typesafe.ai/blog/introducing-system-one-models-and-jev): not another larger transformer, but a shift in architecture and use case, with a speed-to-performance ratio that felt fresh. It is not in the same league as the latest large chat models. Classifiers like this have existed for decades, but a fast general one, cheap enough to steer a control loop at millisecond scale (API roundtrip included), reminded me a little of how the amygdala seems to operate. The paper I have in that direction is V-JEPA 2: a predictive model in representation space that can be pointed at a physical environment and used for control [[8](https://arxiv.org/pdf/2506.09985)].

Anyway, the destination I keep sketching is not one monolithic brain. It is a **sum of parts**: fast and slow thinking, mutable and immutable pieces, shared experts, episodic writes that can consolidate into a parametric feel instead of only retrieving. Some form of Mixture of Experts still looks like fertile ground for that. The hardware is probably the larger limit. Current chips are far from parallel and energy-efficient in the way a brain is, and neuromorphic hardware is not here at scale. Until then, the work is finding ways to organize the transistors we already have, from the software side.

What I am sitting with is still **attention, memory, and routing**: how episodes get written, how they distill, how specialists get woken, how the whole thing stays inside an energy budget. Two open questions for me: what decides when an episodic write is worth consolidating into the weights, and whether a router can ever be more than a soft switch without something closer to a control cortex.

### References

1. [Lennie, P. (2003). The Cost of Cortical Computation.](https://www.cns.nyu.edu/~pl/pubs/Lennie03a.pdf)
2. [Dai et al. (ACL 2024). DeepSeekMoE.](https://aclanthology.org/2024.acl-long.70.pdf)
3. [Geva, Schuster, Berant, and Levy (EMNLP 2021). Transformer Feed-Forward Layers Are Key-Value Memories.](https://aclanthology.org/2021.emnlp-main.446.pdf)
4. [McClelland, McNaughton, and O’Reilly (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex.](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)
5. [Behrouz, Zhong, and Mirrokni (2025). Titans: Learning to Memorize at Test Time.](https://arxiv.org/pdf/2501.00663)
6. [Tandon, Dalal, Li, et al. (2025). End-to-End Test-Time Training for Long Context.](https://arxiv.org/pdf/2512.23675)
7. [Khanna, Kharbanda, Li, Varma, Wang, et al. (2025). Mercury: Ultra-Fast Language Models Based on Diffusion.](https://arxiv.org/pdf/2506.17298)
8. [Assran, Bardes, et al. (2025). V-JEPA 2.](https://arxiv.org/pdf/2506.09985)
