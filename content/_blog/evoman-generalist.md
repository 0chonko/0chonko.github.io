---
title: "Can One EvoMan Agent Beat Eight Enemies?"
date: 2023-10-20
description: "A course-project retrospective on evolving an EvoMan controller from single-enemy specialists to an eight-enemy evaluation."
---

<style>
  .article .evoman-chart { margin: 1.25rem 0; }
  @media (max-width: 420px) {
    .article .evoman-chart img {
      max-width: min(360px, calc(100% + 3rem));
      margin-inline-start: -1.5rem;
    }
  }
</style>

In our Evolutionary Computing course project, we trained neural-network controllers to play EvoMan, a game where an agent faces enemies with different attack patterns. We began by evolving a separate specialist for each opponent. For Task II, we asked how far one controller could get against all eight. This retrospective follows the experiments in Group 55’s October 20, 2023 report.

## From specialists to a diverse population

Our specialist experiments compared two ways to choose who survives each generation. In the comma strategy, `(μ, λ)`, we selected the next generation from the offspring alone. In the plus strategy, `(μ + λ)`, parents competed with offspring for the same places. The plus strategy could carry a strong controller forward unchanged, but if close relatives filled the population, evolution had fewer distinct candidates to build on.

We tried fitness sharing to slow that crowding. Controllers with similar genomes shared credit, reducing the selection advantage of clusters of near-duplicates. Across 10 runs in the reported comparison, fitness sharing sustained genetic diversity longer and improved average fitness. Yet genetically different controllers could still behave similarly in the game. A population could therefore stay diverse in its weights while producing similar moves on screen.

## Training for more than one opponent

We also tried warm initialization, using previously evolved controllers to seed the initial population. This gave evolution a starting point in weights that had already worked in the game. In the report, the warm start improved early fitness.

We then trained with multiple enemies. Each tested training set covered six of the eight opponents. Enemies 3, 5, 7, and 8 appeared in both sets. One set added enemies 1 and 2, while the other added 4 and 6. The chart shows which encounters each training set included.

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/training-coverage.svg' | relative_url }}" alt="Training coverage. Set A includes enemies 1, 2, 3, 5, 7, and 8. Set B includes enemies 3, 4, 5, 6, 7, and 8." />
  <figcaption>Original chart based on the Group 55 Task II report. <a href="{{ '/assets/blog/evoman-generalist/training-coverage.csv' | relative_url }}" download>Download training coverage CSV</a>.</figcaption>
</figure>

## One controller against all eight

In the final evaluation, we measured gain as the agent’s remaining energy minus the enemy’s remaining energy. The best controller reported by Group 55 recorded positive gain against six enemies. Its largest gains came against enemies 5 (+74) and 2 (+68), while the margins against enemies 3 (+4) and 6 (+3) were narrow. It lost to enemies 1 (−90) and 4 (−20). Those eight scores describe a single controller’s evaluation. It handled several opponents, and two still got the better of it.

<figure class="evoman-chart">
  <img src="{{ '/assets/blog/evoman-generalist/enemy-gains.svg' | relative_url }}" alt="Best reported controller’s gains against enemies 1 through 8, respectively: −90, +68, +4, −20, +74, +3, +53, and +24." />
  <figcaption>Original chart based on the Group 55 Task II report. <a href="{{ '/assets/blog/evoman-generalist/enemy-gains.csv' | relative_url }}" download>Download enemy gains CSV</a>.</figcaption>
</figure>
