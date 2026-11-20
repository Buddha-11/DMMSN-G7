# Queue-Aware R-MADDPG Project Backlog

This backlog turns the methodology in `report/IEEEtran/main.pdf`, `dm1.pdf`,
and `dm2.pdf` into implementation gates. A gate is complete only when its
listed verification passes; training curves are not substitutes for simulator
correctness.

## Status legend

- `[x]` implemented and locally verified
- `[~]` started or specified but not complete
- `[ ]` not started

## 0. Method and reproducibility freeze

- [x] Separate the new VEC system from the earlier diffusion-MAPPO/NOMA code.
- [x] Record the task path, queue units, slot causality, deadline policy, and
  physical-vs-virtual energy distinction in `README_PROJECT.md`.
- [x] Audit the supplied papers and current repository for reusable pieces and
  missing contracts.
- [x] Identify mobility data, simulation tools, and reference algorithm code.
- [x] Add one versioned experiment configuration format and save the full
  configuration, seed, code revision, and scenario identifier with every run.
- [x] Create deterministic train/validation/test scenario manifests. Split by
  complete traces/scenarios, never by adjacent slots from one trace.

## 1. Persistent task and queue simulator

**Status: complete.** The gate is covered by deterministic unit tests for
queue conservation, causality, mobility, contact loss, channel rates, and
orthogonal bandwidth allocation.

- [x] Add immutable task specifications with persistent IDs, input bits, CPU
  cycles, arrival timestamps, deadlines, and cycles-per-bit intensity.
- [x] Add ingress, RSU/UAV upload, receive/staging, and local/RSU/UAV compute
  queues with explicit bit/cycle units.
- [x] Express every backlog in equivalent CPU cycles while retaining bit-based
  radio capacities.
- [x] Implement deterministic FIFO radio service and work-conserving weighted
  processor sharing.
- [x] Enforce boundary causality: transferred work cannot receive downstream
  service in the slot that transferred it.
- [x] Keep missed-deadline work in the system and emit one miss event per task.
- [x] Produce per-slot queue arrival/departure ledgers and assert end-to-end
  workload conservation.
- [x] Add analytical tests for persistence, partial transfer, stage causality,
  weighted CPU service, arrival timing, completion time, and deadline handling.
- [x] Add road-constrained synthetic vehicle mobility and distance-dependent
  V2R/V2U rates using the audited channel model.
- [x] Add deterministic queue-aware orthogonal bandwidth allocation shared by
  every controller.

**Gate:** fixed small scenarios match hand calculations; task work is never
created, overwritten, or lost.

## 2. Cloud split, UAV energy, and flight feasibility

**Status: complete.** The environment now accounts for every declared energy
owner, reserves a checked landing trajectory, masks returning/landed UAVs, and
exposes explicit failure rather than accepting an unsafe or underfunded path.

- [x] Add UAV-local/cloud branches with a split fixed at dispatch, including
  `rho=0`, `rho=1`, and indivisible-task cases.
- [x] Add finite UAV-cloud backhaul, finite cloud CPU, branch synchronization,
  and merge overhead.
- [x] Add rotary-wing horizontal propulsion plus declared/calibrated vertical,
  reception, backhaul, computation, control, RSU, cloud, and vehicle energy.
- [x] Add bounded end-of-slot harvesting, physical battery saturation, and
  harvest-overflow logging.
- [x] Add 3-D speed, altitude, acceleration, map, and continuous-segment
  separation checks.
- [x] Keep each accepted motion inside a braking-safe viability set so a UAV
  cannot reach a boundary with insufficient distance to satisfy the next
  slot's acceleration constraint; verified landing arrivals are explicit.
- [x] Add a verified return/landing reserve, destination masking, and an
  explicit physical-infeasibility failure when no contingency exists.
- [x] Add deterministic coupled action repair and log proposals, executions,
  correction magnitudes, and search effort.

**Gate:** split work is conserved, batteries never go negative, energy is
charged to the correct owner exactly once, and no accepted path violates a
flight constraint.

## 3. Lyapunov objective and common control interface

**Status: complete.** Symbolic drift calculations and full environment traces
verify normalized queues, realized-harvest virtual updates, balanced internal
flows, observations, masks, action-repair records, and episode flags.

- [x] Add normalized physical queue state and one virtual net-energy deficit
  queue per UAV.
- [x] Update virtual energy only with realized end-of-slot harvest.
- [x] Compute the per-slot drift-plus-penalty score from actual queue
  increments, service, energy, occupancy, and first deadline crossings.
- [x] Verify that every internal transfer contributes both its source departure
  and destination arrival; use expected harvest only for pre-action scoring.
- [x] Expose role observations, global critic state, masks, proposed actions,
  executed actions, and truncation-vs-terminal flags.

**Gate:** symbolic two-queue examples and numerical slot traces agree with the
implemented score and virtual-queue update.

## 4. Fair non-learning baselines

**Status: complete.** All controllers emit the common `StepAction`, run through
the same masks, scheduler, repair, transition, Lyapunov score, and rollout
metrics. Seed and candidate/runtime budgets are recorded per decision.

- [x] Local-only controller.
- [x] Nearest-feasible edge controller.
- [x] Queue-and-energy-aware greedy controller.
- [x] Genetic/drift optimizer adapted from Q-TOP with fixed evaluation budget.
- [x] Random feasible controller used only as a sanity floor.
- [x] Route every baseline through the same scheduler, feasibility controller,
  simulator, random traces, and metric code.

**Gate:** identical proposed actions produce identical executed transitions
across controllers, and baseline runtime/search budgets are reported.

## 5. Hybrid MADDPG and Reptile meta-learning

**Status: complete.** The PyTorch learning stack is connected to the common
control interface and covered by deterministic checkpoint, replay-isolation,
meta-update, adaptation, masking, and regression tests.

- [x] Implement role-shared vehicle, RSU, and UAV actors with masked task/set
  encoders.
- [x] Use masked categorical hold/destination heads with straight-through
  Gumbel-softmax and continuous split/CPU/motion heads.
- [x] Add centralized critics, task-local replay, target action repair, Polyak
  updates, repair regularization, and gradient diagnostics.
- [x] Implement Reptile over both online actor and critic parameters; reset
  inner optimizer moments and replay for each scenario.
- [x] Synchronize targets at task initialization and after every outer update.
- [x] Add bounded online adaptation that retains physical queues, positions,
  batteries, and task ages across declared change points.

**Gate:** deterministic smoke training saves/reloads exactly, meta-task replay
is isolated, and a toy distribution shows faster matched-budget adaptation than
random initialization without resetting environment state.

## 6. Evaluation and evidence

**Status: evaluation infrastructure complete; final benchmark execution is
pending trained checkpoints.** A hashed smoke-evidence bundle verifies the
full pipeline, but it is deliberately marked as an incomplete comparison and
must not be used for final algorithm-ranking claims.

- [x] Implement low-load, near-capacity, overload, zero-harvest, harvest-outage,
  backhaul-bottleneck, and traffic-surge suites.
- [x] Report response-time mean/p95/p99, deadline success, throughput, every
  stage backlog and growth slope, unfinished/censored tasks, all energy terms,
  battery minima, virtual deficits, landing time, repair rate, and violations.
- [~] Compare local, nearest, greedy, genetic, MADDPG, domain-randomized
  MADDPG, R-MADDPG, queue-weight ablation, and hybrid MAPPO.
  The evaluator enforces this roster and accepts learned checkpoints/common-
  interface controller factories; final learned checkpoints are not yet
  available, so the full comparison has not been claimed.
- [x] Run paired seeds and confidence intervals on shared traffic/channel/
  harvest realizations.
- [x] Measure zero-update generalization, adaptation curve, recovery time,
  update/inference/coordination latency, and offline training cost.
- [x] Drain finite traces or explicitly report censored unfinished work.

**Gate:** every headline claim is backed by a saved trace, configuration,
metric table, uncertainty estimate, and reproducible command.

## 6.5 Reproducible learned-policy training

**Status: orchestration implemented; research-scale run safely paused.** The trainer
keeps the test split sealed, selects checkpoints on held-out validation reward,
rejects safety-infeasible validation checkpoints, and records wall-clock cost
and exact resume state. A tiny smoke run is an integration check, not evidence
of convergence or model quality.

- [x] Add single-stress hybrid MADDPG, domain-randomized MADDPG, and Reptile
  inner/outer training modes.
- [x] Add bounded hybrid-action exploration that retains categorical masks,
  continuous limits, and CPU-share normalization.
- [x] Add validation-only checkpoint selection, early stopping, best/last
  checkpoints, exact RNG/replay/optimizer resume state, and JSONL curves.
- [x] Add the physical-queue-weight ablation without weakening battery or
  flight feasibility enforcement.
- [ ] Execute preregistered multi-seed research-scale training and retain the
  complete checkpoint/provenance bundles.
- [x] Implement and register an independent on-policy hybrid-MAPPO comparison
  with role-shared stochastic actors, centralized value learning, GAE, clipped
  PPO, entropy/KL controls, and executed-action repair regularization.
- [x] Add a resumable multi-seed study driver that freezes one scenario
  manifest, trains every learned method for every learner seed, enforces the
  complete comparison roster, and aggregates uncertainty at the independent
  learner-seed level after averaging paired test scenarios.
- [ ] Execute research-scale hybrid-MAPPO training with the same held-out
  checkpoint-selection and independent-seed protocol as the MADDPG families.

**Gate:** every final learned method has independent-seed checkpoints selected
without test data, learning curves, offline cost, and successful held-out test
evaluation through the common Phase 6 runner.

The current two-epoch/default-network domain-randomized MADDPG pilot performed
117 gradient updates and produced a safety-eligible checkpoint. The separate
two-epoch/default-network MAPPO pilot performed six on-policy scenario updates
and also produced a safety-eligible checkpoint. Both are short engineering
pilots; their one-seed test bundles are explicitly incomplete and must not be
used for a paper-level algorithm ranking.

The complete study-driver smoke run under `output/studies/smoke-v1` produced
all five learned checkpoints and a paired 9-method/7-scenario bundle (63 runs)
with no physical, battery, or separation violations. It is explicitly marked
as smoke evidence. The five-seed research plan is frozen under
`output/studies/research-v1`. It was safely paused on 2026-10-28 with 3 of 25
training jobs complete and four partial jobs checkpointed at epochs 4, 23, 11,
and 35. `study_pause.json` records the verified resume state and exact command;
the sealed test evaluation has not begun, so neither pending research-scale
execution item is complete.

## 7. Report completion

- [ ] Replace placeholder benchmark rows in the IEEE report with implemented
  baselines and exact configurations.
- [ ] Add numerical results, ablations, overload/feasibility failures, compute
  cost, and limitations.
- [ ] Reconcile all equations, action definitions, timing, and claimed
  guarantees with code behavior.
- [ ] Publish an artifact manifest with dataset licenses, code dependencies,
  seeds, checkpoints, and result hashes.

## Data and tools selected by the web audit

### Required

- **Python, NumPy, and PyTorch:** simulator/learning stack. The first simulator
  milestone uses no new third-party dependency beyond the repository baseline.
- **SUMO plus libsumo/TraCI:** main vehicle mobility source. SUMO is a
  microscopic traffic simulator; its official docs recommend libsumo when
  performance matters and document subscription-based retrieval of vehicle
  positions: <https://sumo.dlr.de/docs/TraCI/>.
- **Bologna iTETRIS scenario:** primary compact real-road evaluation because it
  is the Bologna family used by Q-TOP and is listed in SUMO's official scenario
  catalog: <https://sumo.dlr.de/docs/Data/Scenarios.html>.
- **Synthetic task traces:** required for controlled low/near-capacity/overload
  experiments. Generate them from versioned distributions matching the paper
  parameter ranges; no external workload dataset is required for the core
  hypothesis.

### Recommended robustness data

- **TAPAS Cologne:** large 24-hour SUMO scenario. Its official documentation
  warns about mapping/network quality and requires attribution, notification,
  and non-commercial use under the stated terms. Use it only after recording
  those conditions: <https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html>.
- **OSM-generated Qingdao/Pasubio subsets:** use SUMO's OSM Web Wizard for
  reproducible scenario generation rather than treating figures or Google Maps
  imagery as trajectory data: <https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html>.
- **Alibaba or Google cluster traces (optional):** useful only for a separately
  documented workload-shape sensitivity study; they do not directly provide
  vehicular input bits, CPU cycles, radio contacts, or deadlines. Do not call
  them a VEC dataset. Sources: <https://github.com/alibaba/clusterdata> and
  <https://github.com/google/cluster-data>.

### Reusable algorithm references

- OpenAI's MADDPG reference is archived, TensorFlow-era code. Reuse the
  centralized-training structure and equations, not its obsolete runtime:
  <https://github.com/openai/maddpg>.
- OpenAI's supervised Reptile reference cleanly demonstrates the outer update,
  but it is not an RL/offloading implementation:
  <https://github.com/openai/supervised-reptile>.
- PettingZoo's Parallel API is a good later interoperability wrapper for
  simultaneous role actions, but should wrap the validated simulator rather
  than define its physics: <https://pettingzoo.farama.org/api/parallel/>.
- SciPy's global optimizers can support the Q-TOP-style search baseline after
  action encoding and constraints are fixed:
  <https://docs.scipy.org/doc/scipy/reference/optimize.html>.

The targeted search found no public repository claiming to implement either
supplied 2026 paper. Third-party UAV-MEC repositories can inform test cases, but
their environment assumptions must not be mixed into the benchmark core.
