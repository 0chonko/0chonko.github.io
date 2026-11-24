---
title: "Can One EvoMan Agent Beat Eight Enemies?"
date: 2023-10-20
description: "A course-project retrospective on evolving an EvoMan controller from single-enemy specialists to an eight-enemy evaluation."
---

<style>
  .article .evoman-chart { margin: 1.25rem 0; }
  @media (max-width: 600px) {
    .article .evoman-steps td:first-child { white-space: normal; }
    .article .evoman-steps th,
    .article .evoman-steps td {
      padding: 0.4rem 0.25rem;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
  }
  @media (max-width: 420px) {
    .article .evoman-chart img {
      max-width: min(360px, calc(100% + 3rem));
      margin-inline-start: -1.5rem;
    }
  }
</style>

In our Evolutionary Computing course project, we trained neural-network controllers to play EvoMan, a game where an agent faces enemies with different attack patterns. Each controller encoded 265 weights and biases for a network with 20 sensor inputs, 10 sigmoid hidden units, and five action outputs. We began by evolving a separate specialist for each opponent. For Task II, we asked how far one controller could get against all eight. This retrospective follows the experiments in Group 55’s October 20, 2023 report.

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/gameplay-enemy-2.webp' | relative_url }}" alt="EvoMan gameplay with the blue player jumping near a projectile, enemy 2 on the right, and both energy bars above the arena." />
  <figcaption>A local replay against enemy 2 using the saved project controller. The top bars show player and enemy energy.</figcaption>
</figure>

## From specialists to a diverse population

We used DEAP to evolve the weight vectors. Each generation produced λ offspring and selected μ survivors. In the comma strategy, `(μ, λ)`, we selected the next generation from the offspring alone. In the plus strategy, `(μ + λ)`, parents competed with offspring for the same places. The plus strategy could carry a strong controller forward unchanged, but if close relatives filled the population, evolution had fewer distinct candidates to build on.

We tried fitness sharing to slow that crowding. We measured Euclidean distance between weight vectors. A neighbor within the sharing radius contributed a penalty that decreased linearly with distance, reducing the selection advantage of clusters of near-duplicates. Across 10 runs in the reported comparison, fitness sharing sustained genetic diversity longer and improved average fitness. Yet genetically different controllers could still behave similarly in the game. A population could therefore stay diverse in its weights while producing similar moves on screen.

## What changed from Task I to Task II

<div class="table-wrap evoman-steps" markdown="1">

| Step | What changed | Reported result |
| --- | --- | --- |
| 1. Specialists | Evolved a controller per enemy to compare offspring-only selection with selection from parents and offspring. | The comma strategy retained more diversity. The plus strategy converged earlier and became the basis for Task II. |
| 2. Sharing | Subtracted a distance-based penalty for neighboring genomes. The Task II settings list a sharing radius of 0.01. | Across 10 runs, diversity lasted longer and average fitness improved. Agents with different weights still showed similar behavior. |
| 3. Warm start | Seeded each stage with winners from runs on smaller enemy sets. | Seeded controllers showed stronger fitness from the first generation. |
| 4. Generalist | Trained on groups of enemies and scored both average performance and consistency. The DEAP strategy also applied enemy-specific fitness biases. | The plus strategy produced the best individual in the reported comparison. Its gains across all eight enemies appear below. |

</div>

Task II’s DEAP strategy used simulated binary bounded crossover and polynomial bounded mutation. The custom comparison used tournament selection, uniform crossover, Gaussian mutation, and an elitist survivor step. It increased the mutation standard deviation when fitness stalled and reset it after an improvement. The report lists population sizes of 30, 44, and 137 and used Optuna with CMA-ES and Hyperband for tuning.

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/training-workflow.png' | relative_url }}" alt="The report’s workflow starts parallel specialist populations on individual enemies, pools winners to seed runs on progressively larger sets, and finishes with fitness sharing and random enemy sampling." />
  <figcaption>Training workflow from the Group 55 Task II report. FS denotes fitness sharing, SEL selection, CX crossover, and MUT mutation.</figcaption>
</figure>

## Training for more than one opponent

We also tried warm initialization, using previously evolved controllers to seed the initial population. Each stage saved its best controllers to seed the next, moving from one enemy to samples of four or five before the final stage. In the report, the warm start improved early fitness.

We then trained with multiple enemies. EvoMan’s multi-enemy fitness used the mean of the per-enemy fitness scores minus their standard deviation, rewarding controllers that performed consistently across encounters. Each tested training set covered six of the eight opponents. Enemies 3, 5, 7, and 8 appeared in both sets. One set added enemies 1 and 2, while the other added 4 and 6. The chart shows which encounters each training set included.

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/training-coverage.svg' | relative_url }}" alt="Training coverage. Set A includes enemies 1, 2, 3, 5, 7, and 8. Set B includes enemies 3, 4, 5, 6, 7, and 8." />
  <figcaption>Original chart based on the Group 55 Task II report. <a href="{{ '/assets/blog/evoman-generalist/training-coverage.csv' | relative_url }}" download>Download training coverage CSV</a>.</figcaption>
</figure>

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/gameplay-enemy-5.webp' | relative_url }}" alt="EvoMan gameplay with the player on the left, enemy 5 on the right, and projectiles crossing the arena." />
  <figcaption>A local replay of the same saved controller against enemy 5.</figcaption>
</figure>

## One controller against all eight

In the final evaluation, we measured gain as the agent’s remaining energy minus the enemy’s remaining energy. The best controller reported by Group 55 recorded positive gain against six enemies. Its largest gains came against enemies 5 (+74) and 2 (+68), while the margins against enemies 3 (+4) and 6 (+3) were narrow. It lost to enemies 1 (−90) and 4 (−20). Those eight scores describe a single controller’s evaluation. It handled several opponents, and two still got the better of it.

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/enemy-gains.svg' | relative_url }}" alt="Best reported controller’s gains against enemies 1 through 8, respectively: −90, +68, +4, −20, +74, +3, +53, and +24." />
  <figcaption>Original chart based on the Group 55 Task II report. <a href="{{ '/assets/blog/evoman-generalist/enemy-gains.csv' | relative_url }}" download>Download enemy gains CSV</a>.</figcaption>
</figure>
