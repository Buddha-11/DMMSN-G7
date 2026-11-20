# Domain-Randomized MADDPG Report Update Plan

## Objective

Create a separate IEEE-style report whose proposed method is queue-aware,
domain-randomized Hybrid MADDPG. The existing Reptile report remains unchanged.

## Writing and formatting pattern

The new report follows the supplied MAPPO paper's progression:

1. Introduction and explicit contributions.
2. Related work organized by technical theme.
3. Detailed system model and constrained optimization problem.
4. Proposed methodology with MDP definition, CTDE actor-critic equations,
   training phase, pseudocode, and complexity analysis.
5. Numerical results beginning with simulation parameters and baselines,
   followed by convergence, performance, stress, runtime, safety, and ablation
   evidence.
6. Conclusion with claims restricted to completed experiments.

## Algorithmic changes

- Rename the proposed approach from R-MADDPG to QA-DR-MADDPG.
- Remove Reptile inner/outer meta-learning from the proposed method.
- Replace the meta-objective by an expectation over randomized domains.
- Preserve the hybrid actor, centralized critic, target networks, replay,
  straight-through categorical decisions, action repair, and CTDE execution.
- Preserve physical queues, virtual energy queues, and the Lyapunov reward.
- Replace online few-shot adaptation claims with zero-update generalization.
- Keep Reptile-MADDPG and MAPPO only as learned comparison methods.

## Sections retained with limited edits

- Network and time-slot model.
- Mobility and coverage model.
- Task generation, routing, partitioning, and deadline model.
- Air-to-ground and terrestrial channel models.
- Transmission, computation, backhaul, and persistent queue models.
- UAV flight, hovering, communication, computation, and control energy.
- Harvesting, battery causality, return reserve, and collision avoidance.
- Long-term constrained objective and drift-plus-penalty transformation.

The detailed system model is restored from the professor-approved original
report in `restored_system_model.tex`. It preserves the original end-to-end
service graph, principal-notation table, slot chronology, queue-conservation
model, separated flight/hover energy ledger, optional laser-transfer model,
and return-reserve constraints. These models are independent of whether the
policy is trained with Reptile or domain randomization.

## Numerical-results changes

- Exclude local-only, nearest-edge, queue-greedy, genetic, and random methods
  from the report figures and comparison tables.
- Compare QA-DR-MADDPG with Hybrid MADDPG, Reptile-MADDPG, queue-weight
  Reptile ablation, and Hybrid MAPPO.
- Add a simulation-parameter table using the frozen research-v1 configuration.
- Add overall performance, stress-condition, and learning-curve figures.
- Add a domain-randomization ablation: Hybrid MADDPG versus QA-DR-MADDPG.
- Report the available Reptile queue-weight ablation only as supporting
  evidence; do not mislabel it as an ablation of QA-DR-MADDPG.
- State that a direct domain-randomized no-queue run is required before the
  queue contribution can be isolated on the proposed backbone.
- Report five-seed uncertainty and avoid superiority claims where paired
  confidence intervals cross zero.

## Verification

- Generate all figures directly from frozen CSV/history artifacts.
- Compile with the repository's IEEEtran class.
- Resolve LaTeX errors and references.
- Inspect the final PDF text and page renders for clipped tables, missing
  figures, placeholder text, and unsupported claims.
