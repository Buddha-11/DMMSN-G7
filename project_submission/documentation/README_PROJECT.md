# Queue-Aware Meta-Reinforcement Learning for Multi-UAV Assisted Vehicular Edge Computing

## 1. Project purpose and status

This project implements a simulation and learning framework that jointly controls task offloading, computing-resource allocation, and three-dimensional UAV movement in a changing vehicular network. Its goal is to reduce task response time and energy consumption while keeping workloads manageable and respecting finite UAV batteries replenished by uncertain energy harvesting.

The project combines the adaptive learning mechanism of R-MADDPG with queue-aware vehicular modeling and Lyapunov optimization. The proposed controller is called **queue-aware R-MADDPG** in this document. This is a descriptive working name, not a claim of an already validated algorithm.

**Status:** the queue-aware VEC simulator, feasibility layer, MADDPG/Reptile learners, independent hybrid-MAPPO comparison, training orchestration, and common evaluation pipeline are implemented and covered by automated tests. Only short engineering pilots have been run; research-scale, independent-seed training and the final complete comparison remain pending. The earlier Diffusion-MAPPO code is retained separately and is not used as evidence for this framework.

This document resolves the design ambiguities in `README_MERGED.md` and the conceptual draft in `report/IEEEtran/main.tex`. Where the source papers leave a gap, the choices below are proposed engineering and research decisions rather than results borrowed from those papers.

## 2. The problem in plain language

Vehicles continually generate computational tasks. A vehicle may process a task itself or upload it to an RSU or a UAV. A UAV can execute uploaded work and forward a selected portion to the cloud.

The closest server is not necessarily the fastest destination. A nearby UAV might have a congested queue, insufficient battery, or too little CPU capacity. Another UAV or an RSU could finish the task earlier despite a longer upload. Moving UAVs toward demand can improve connectivity, but consumes energy and changes coverage for other vehicles.

The controller must therefore consider the whole chain: movement affects communication; communication affects where tasks arrive; those arrivals affect queues; CPU service affects waiting time and energy; battery availability affects future movement and service.

The research question is:

> Can a meta-trained multi-agent controller adapt to new vehicular workloads and operating conditions while achieving a useful latency-energy trade-off, controlling task backlogs, and obeying physical energy and flight constraints?

## 3. What comes from the two papers

| Component | Source | How this project uses it |
|---|---|---|
| UAV/RSU/local execution and UAV-to-cloud forwarding | [dm1.pdf](dm1.pdf), Sections III-IV | Execution destinations and a UAV/cloud split |
| Per-user computing-resource allocation | dm1 | CPU allocation under server capacity constraints |
| MADDPG with Reptile meta-learning | dm1 | An inner adaptation loop and an outer initialization-learning loop |
| Vehicle, RSU, and UAV computation queues | [dm2.pdf](dm2.pdf), Section III | Persistent workload and waiting-time modeling |
| Speed, azimuth, and elevation control | dm2 | Three-dimensional UAV movement |
| Propulsion, computation, harvesting, and energy budget | dm2 | A finite-battery energy model with uncertain replenishment |
| Lyapunov transformation | dm2, Section IV | A queue-weighted per-slot optimization objective |
| Genetic per-slot optimization | dm2, Section V | A comparison method for the same proposed environment |

The integration adds transmission queues, explicit battery feasibility, an energy virtual queue, a defined hybrid action mechanism, and a common evaluation protocol. Paper 1 does not put movement in its stated action space. Q-TOP uses a genetic algorithm rather than meta-RL, and its displayed Lyapunov function contains task queues rather than an explicit energy virtual queue.

These differences motivate the project. Establishing novelty beyond the two supplied papers requires a broader literature review.

## 4. System architecture and scope

The network contains mobile vehicles, fixed RSUs, mobile UAV edge servers, and a remote cloud server. Time is divided into slots of duration `Delta` seconds. Mobility and channel conditions are updated between slots; capacities within a slot use a documented piecewise-constant approximation.

```text
Vehicle task arrival
        |
        +--> Local computation queue --> completion
        |
        +--> Vehicle-to-RSU upload --> RSU computation queue --> completion
        |
        +--> Vehicle-to-UAV upload
                    |
                    +--> UAV computation queue ------+
                    |                                |
                    +--> UAV-cloud backhaul           +--> task completion
                              --> Cloud computation --+

Queue and battery observations --> Hybrid R-MADDPG actions
                               --> Feasibility controller
                               --> Physical execution and queue updates
```

RSUs and UAVs are alternative destinations. There is no mandatory vehicle-to-RSU-to-UAV chain. The initial model has no direct vehicle-to-cloud route, RSU-to-cloud route, inter-UAV task migration, or repeated destination changes after dispatch.

Core experiments use orthogonal radio allocation with explicit finite bandwidth and no jammers. Diffusion actors, NOMA, cooperative caching, and jammer avoidance from the existing repository are optional later extensions. They are not needed to test the central queue-aware meta-learning hypothesis.

## 5. Tasks, routing, and completion

Each task `i` records:

| Variable | Meaning | Unit |
|---|---|---|
| `b_i` | Input data size | bits |
| `c_i` | Total CPU work | cycles |
| `chi_i = c_i / b_i` | Computation intensity | cycles/bit |
| `t_i_arr` | Arrival timestamp | seconds |
| `d_i` | Allowed response time | seconds |
| `rho_i` | Fraction executed at its selected UAV | dimensionless |

One destination is selected when the task is dispatched: local, one RSU, or one UAV. Unscheduled tasks remain in the vehicle's ingress buffer. They are never overwritten by the next arrival.

For a UAV destination, the entire input is uploaded to that UAV. The UAV executes `rho_i c_i` cycles and forwards `(1-rho_i)b_i` bits for `(1-rho_i)c_i` cloud cycles. The split is fixed at dispatch. Fractions of both work and data must appear in every corresponding latency and energy calculation.

Only tasks declared partitionable support this split. For indivisible tasks, use `rho_i = 1` initially. Partitionable tasks are represented by bounded chunks; a chunk becomes executable only after its required input has arrived. Chunk granularity is recorded and tested for sensitivity. Final completion waits for every required branch, including a configured merge overhead.

The simulator records actual completion timestamps. Response time is `t_i_done - t_i_arr`, including ingress waiting, upload, server waiting, computation, backhaul, and merge overhead. The core model assumes small result messages and negligible return time; experiments with non-negligible outputs must add a return-link queue.

Deadline misses are recorded when unfinished tasks first exceed their deadlines. Missed tasks continue processing in the core model. This avoids apparent stability caused merely by deleting work. A later expiry policy must report discarded work separately.

## 6. Queue dynamics and unit consistency

Maintain task/chunk records for ingress, vehicle upload, local computation, RSU computation, UAV computation, UAV-cloud backhaul, and cloud computation. Radio queues retain remaining bits; compute queues retain remaining cycles. A task identifier and branch identifier connect all stages.

For the Lyapunov objective, measure every workload queue in **CPU-cycle equivalents**. A communication chunk with `b` remaining bits and intensity `chi` contributes `chi b` equivalent cycles. This makes queue transfers conserve work. The radio capacity constraint remains in bits: transmitting `x` equivalent cycles consumes `x/chi` bits. Different task intensities must be handled per chunk, not through a single assumed global conversion factor.

For each queue `j`, define:

```text
Q_j(t+1) = Q_j(t) - S_j(t) + A_j(t)
0 <= S_j(t) <= Q_j(t)
```

`S_j` is actual completed service or transfer, not nominal CPU/radio capacity. `A_j` includes external arrivals and transfers from upstream queues. Transfers become available downstream at the next slot boundary; a chunk cannot be uploaded, forwarded, and executed using the same slot's full capacity at every stage.

An upstream transfer removes work from one queue and adds the same work to its downstream queue. Only final computation removes that branch's remaining work from the processing network. Splitting distributes work between branches without duplication. Completed branches have zero remaining work even while another branch of the same task is unfinished.

The compute discipline is preemptive weighted processor sharing, enabling meaningful per-task CPU allocation. It deliberately extends Q-TOP's FIFO model. The FIFO version is an evaluation variant, not an interchangeable assumption in the same equations.

Strong stability is the desired condition:

```text
limsup_T (1/T) sum_t E[Q_j(t)] < infinity
```

This can be feasible only when arrivals lie within the achievable radio, computing, and energy capacity region. Under sustained overload, the project measures backlog growth and service failure rather than promising stability. No admission rejection is used to hide overload in the core experiments.

## 7. Communication and computing resources

Vehicle-to-UAV gains depend on distance, elevation, and probabilistic LoS/NLoS attenuation. Vehicle-to-RSU gains use a terrestrial path-loss model. The UAV-cloud link is an effective finite-capacity backhaul; the cloud is not modeled as a radio node at an arbitrary physical altitude.

```text
R_link = B_link log2(1 + P_tx g_link / (N0 B_link + I_link))
sum allocated bandwidth <= node bandwidth budget
```

Here `N0` is noise power spectral density. The core configuration sets `I_link = 0` using orthogonal spectrum across interfering cells; reuse experiments must calculate inter-cell interference explicitly. Fixed transmit powers isolate routing, CPU, and movement decisions initially.

For each compute server `j`, let `f_j` be its active CPU frequency and `w_ij` nonnegative shares summing to at most one. Allocated service is bounded by `Delta f_j w_ij` cycles and remaining eligible work. Local vehicles use a fixed CPU frequency initially; RSUs and UAVs control frequency and shares. The cloud has finite configurable CPU capacity and a fixed scheduler.

Server dynamic power is modeled as `kappa_j f_j^3` while active. Equivalently, serving `s` cycles at frequency `f_j` consumes `kappa_j f_j^2 s` joules. Charge aggregate server energy once, with explicit active time; do not assign full server capacity or full energy independently to every concurrent task.

## 8. UAV motion and finite batteries

UAV `k` chooses speed `v_k`, azimuth `theta_k`, and elevation `phi_k`:

```text
p_k(t+1) = p_k(t) + Delta v_k
           [cos(theta_k)cos(phi_k), sin(theta_k)cos(phi_k), sin(phi_k)]
```

Enforce speed, climb/descent, altitude, acceleration, map boundary, and inter-UAV separation limits. Check minimum separation throughout the piecewise-linear slot trajectories, not only at their endpoints. Use a rotary-wing horizontal propulsion model plus an explicit calibrated vertical-motion correction; a horizontal-only power equation is insufficient for arbitrary climbing.

Battery constraints apply to propulsion, computation, receiver electronics, and UAV-cloud transmission. Vehicle uplink transmit energy belongs to the vehicle, not the receiving UAV. RSU and cloud energy can contribute to the system cost, but do not drain a UAV battery.

Harvest `H_k(t)` arrives at the end of the slot. The controller observes current battery and harvest statistics, not the future realized harvest:

```text
E_k(t) <= B_k(t) - B_k_reserve(t)
B_k(t+1) = min(B_k_max, B_k(t) - E_k(t) + H_k(t))
```

The reserve covers a verified safe return/landing option. An active UAV's fallback is not automatically hovering: hovering also consumes energy. If service is no longer feasible, the controller initiates the feasible return/landing procedure and removes that UAV from available destinations. Landed UAVs do not accumulate airborne propulsion cost.

Start with bounded uniform energy arrivals as a controlled abstraction inspired by Q-TOP. Follow with intermittent and temporally correlated harvesting. Laser harvesting is a motivating mechanism; position-dependent illumination, receiver alignment, and transfer losses are required before claiming a realistic laser-power deployment.

## 9. Long-term objective and Lyapunov integration

Use a dimensionless per-slot cost:

```text
C_t = w_L (Delta N_unfinished(t) / L_ref)
    + w_E (E_system(t) / E_ref)
    + w_D N_new_deadline_misses(t)
```

`N_unfinished` counts tasks, including tasks waiting in ingress or transport. Its time integral is aggregate task sojourn time over a drained finite trace. Under stationary stable operation with fixed arrival rate, it is related to mean response time through Little's law. It supplies a dense latency signal and avoids rewarding a controller that completes only easy tasks. Evaluation still uses measured response times and reports unfinished work.

Fix positive references `L_ref` and `E_ref` from training configuration. Do not rescale each evaluation scenario independently. `w_L`, `w_E`, and `w_D` are declared preference weights.

Require physical battery feasibility and task-queue stability, and impose a long-term net energy target:

```text
limsup_T (1/T) sum_t E[E_k(t) - H_k(t)] <= ebar_k
Z_k(t+1) = max(0, Z_k(t) + E_k(t) - H_k(t) - ebar_k)
```

For indefinite energy-neutral operation, choose `ebar_k = 0`. A positive value represents permitted average battery draw over a finite mission; it is not indefinite sustainability. `Z_k` is a virtual accounting queue and must not be confused with the physical battery.

Use fixed reference scales to define `q_j = Q_j/Q_ref` and `z_k = Z_k/E_qref`. The Lyapunov function is:

```text
L(t) = 0.5 sum_j q_j(t)^2 + 0.5 sum_k z_k(t)^2
```

For bounded arrivals and service, expanding the squared queue updates yields a drift upper bound with a decision-independent constant plus queue-weighted increments. The proposed per-slot control score is:

```text
G_t = V C_t
    + sum_j q_j(t) [A_j(t) - S_j(t)] / Q_ref
    + sum_k z_k(t) [E_k(t) - H_k(t) - ebar_k] / E_qref
reward_t = -G_t
```

Use conditional expected harvest for pre-action candidate scoring. The realized harvest can be used in the transition reward and virtual queue update. With exogenous action-independent harvesting, its contribution is constant across feasible actions in the same state.

Transfers must appear on both sides: moving `x` cycles from queue `a` to `b` contributes `(q_b-q_a)x/Q_ref`. This discourages unloading a vehicle by simply congesting an edge server. External arrivals independent of the action may be omitted from candidate comparisons, but remain in physical queue updates.

`V` controls the relative cost/backlog trade-off. Larger values prioritize the immediate cost more strongly; practical effects are evaluated across a sweep. Because this project uses normalized queues and a different coefficient convention from Q-TOP, its tuning values cannot be copied directly.

The RL controller approximates this constrained per-slot problem. Discounted actor-critic learning does not itself prove average-cost optimality or queue stability. Classical bounds require feasible stabilizable arrivals, suitable moment/slack assumptions, and a bounded approximation error to the per-slot drift minimizer. Establishing such an error for the learned controller is separate research work. Until then, report empirical stability and constraint enforcement, not inherited Q-TOP guarantees.

## 10. Agents, observations, and hybrid actions

Use three actor roles with shared parameters within each role:

| Actor role | Observation | Action |
|---|---|---|
| Vehicle | Pending tasks, own CPU/queue, location, visible node capacity/backlog/battery summaries and channel estimates | Dispatch/hold, categorical destination, UAV/cloud fraction for a UAV destination |
| RSU | Eligible task features, own queue and CPU state | CPU frequency and task shares |
| UAV | Own queue/battery/virtual energy queue, task features, nearby vehicles/UAVs, channel estimates | CPU frequency/shares and 3-D movement |

This explicitly adds vehicle dispatch actors to Paper 1's UAV/RSU agent design. It gives each destination choice one owner and prevents multiple UAVs from independently claiming the same task. Shared vehicle actors limit parameter growth. Actors use masked sets or bounded padded task/node lists, with overflow retained in real buffers and exposed through backlog summaries.

Training critics observe the joint state and joint actions. Execution actors use their role observations plus advertised node summaries. A coordinating feasibility controller is required; the resulting execution is distributed decision-making with coordination, not communication-free decentralized control. Measure the summary-exchange and arbitration overhead.

Destination and dispatch actions are categorical. Use masked categorical logits with straight-through Gumbel-softmax during training and hard argmax during evaluation. Continuous heads produce bounded movement, frequency, and split values; masked softmax produces CPU shares. This is a **hybrid extension of MADDPG**, not unchanged continuous-action MADDPG. Straight-through gradients are biased approximations and must be evaluated against a hybrid policy baseline.

The feasibility controller masks unreachable or unavailable destinations, scales CPU/bandwidth allocations to capacities, and repairs proposed motion and service to meet energy and flight constraints. It must have a deterministic priority/tie-break rule. An upload interrupted by lost coverage remains buffered for its original destination; it is not silently reassigned.

Motion repair also preserves braking distance to the map and altitude bounds.
Endpoint clamping alone is insufficient: it can leave a UAV at a boundary with
outward velocity and no acceleration-feasible command in the following slot.
Only a verified return plan may use the explicit minimum-altitude landing
exception.

Train critics on executed actions, and store both proposals and executed actions. Continuous differentiable constraints can use differentiable projection. Discrete masks and coupled safety repairs use an explicitly documented surrogate gradient or stop-gradient treatment. Add a proposal-to-execution correction loss and report correction frequency. This reduces reliance on repair while acknowledging that the actor gradient is approximate.

## 11. Reptile training and adaptation

A meta-task is an environment distribution, not an individual vehicle computation task. It specifies traffic trace/density, arrival intensity, task sizes and CPU intensities, server capacities, channels, harvesting, and initial energy.

1. Sample a batch of training meta-tasks.
2. For each meta-task, copy the same meta actor/critic initialization.
3. Collect transitions and perform a fixed number of inner hybrid-MADDPG updates using task-specific replay.
4. Average the resulting parameter changes and update the meta initialization:

```text
theta_meta <- theta_meta + epsilon mean_m(theta_m_adapted - theta_meta)
```

5. Repeat, selecting hyperparameters on separate validation scenarios.

Actor and critic weights are meta-trained. Reset inner optimizer moments per sampled task; do not average Adam states. Initialize target networks from the copied online networks, update them within adaptation, and synchronize them after the outer update. Keep replay task-specific to avoid mixing incompatible transition distributions.

At deployment, retain real queues and batteries while collecting a bounded adaptation window and applying a fixed update budget. Changing the environment must not reset backlog to manufacture an adaptation advantage. Reptile improves initialization through training; it does not automatically detect environmental changes. Begin with declared change points, then evaluate a specified detector based on workload/channel statistics.

Domain randomization alone is a distinct baseline: it trains one policy on diverse environments without learning through inner/outer adaptation.

## 12. Per-slot execution order

1. Read current queues, batteries, vehicle/UAV positions, and eligible tasks; receive node summaries.
2. Produce routing, split, CPU, bandwidth-scheduler, and movement proposals.
3. Apply routing masks and coupled physical feasibility checks; record corrections.
4. Execute communication and computation on work available at slot start, and fly along the feasible trajectory.
5. Commit service and stage transfers at the boundary; retain every unfinished chunk.
6. Add external task arrivals and realized harvested energy; update task queues, batteries, and virtual energy queues.
7. Record task completion/miss events, cost, reward, and the full transition.
8. Apply scheduled inner adaptation updates without altering the physical state.

Bandwidth is allocated by a deterministic queue-aware scheduler initially. It is an explicit supporting mechanism shared by all algorithms; bandwidth optimization is not claimed as an additional learned contribution.

## 13. Existing code and implementation plan

Inspection of `src/env.py` shows a single `current_task` per device regenerated each slot, a declared UAV `task_queue` list, direct task-delay calculations, and a shaped latency/energy reward. These are not the persistent queue network above. Its five-dimensional UAV action contains movement and a device scheduling hint; it does not implement the proposed complete routing/CPU/split controls.

`src/baselines/maddpg_agent.py` already provides deterministic actors, centralized critics, replay, target networks, and off-policy updates. This is reusable infrastructure, but needs role-specific hybrid actions and the meta loop. `DomainRandomiser`, channel models, metrics, and logging are also reusable after their assumptions are audited.

Suggested new modules below are planned files, not existing capabilities:

| Phase | Planned work | Completion criterion |
|---|---|---|
| 1 | `src/vec/tasks.py`, `queues.py`, `env.py`; local/RSU/UAV processing and traffic | Persistent task IDs, conserved work, real waiting times, reproducible traces |
| 2 | Backhaul/cloud stages, `energy.py`, `safety.py` | Split conservation, correct energy ownership, nonnegative finite batteries, feasible fallback and trajectories |
| 3 | `lyapunov.py` and common scheduler | All stage increments enter the objective; virtual energy queues update correctly |
| 4 | Queue-aware greedy and genetic per-slot baselines | All methods use identical dynamics, constraints, costs, and information |
| 5 | `hybrid_maddpg.py`, `train_meta.py` | Valid role actions, reproducible inner/outer updates, saved adaptation configuration |
| 6 | `evaluate_vec.py` and result plots | Held-out adaptation, backlog, energy, latency, and computational overhead results |
| 7 | Update `report/IEEEtran/main.tex` | Equations and claims agree with implemented behavior and measured evidence |

Keep the earlier DMJO environment and entry points available for reproducibility. Implement the new environment separately and move only audited common utilities into shared modules. Existing DMJO results do not validate the proposed VEC system.

## 14. Evaluation and evidence required

Use road-constrained synthetic mobility for initial checks and SUMO traces for the main vehicular experiments. Split training, validation, and testing by traces/scenarios, not by adjacent time slots of one trace. First vary density, demand, capacities, channel conditions, and harvest statistics with a fixed topology. Vary node counts only after masked/shared encoders support them consistently.

Compare local-only, nearest-feasible edge, queue-and-energy-aware greedy, a genetic drift optimizer adapted from Q-TOP, hybrid MADDPG without meta-learning, domain-randomized hybrid MADDPG, R-MADDPG without queue weighting, and a hybrid MAPPO baseline. Apply the same physical feasibility controller to every method. Report learned policy quality separately from the benefits of that controller.

Ablate meta-learning, task drift weighting, energy virtual queues, learned movement, CPU allocation, and cloud splitting individually. Removing an energy virtual queue never removes physical battery limits.

Report mean and tail response time, deadline success rate, completed-task throughput, task/work backlog at every stage, backlog growth slopes, unfinished tasks, UAV and total system energy, battery minima, virtual energy backlog, landing/unavailability time, harvest overflow, and safety correction rates. Report actual collisions or battery violations as failures.

Measure adaptation before and after changes under matched environment-interaction and gradient-update budgets. Include zero-update generalization, adaptation curves, recovery time, online update cost, inference/coordination latency, and offline training cost. Use multiple independent seeds and confidence intervals, pairing algorithms on the same traffic and harvest realizations.

Long rollouts must retain queues and energy rather than restarting periodically. Drain finite evaluation traces without new arrivals to measure remaining completion times, or report censored tasks explicitly. Never report only successful-task latency without completion and backlog statistics.

Check feasibility under low, near-capacity, and overloaded workloads, including zero harvesting, prolonged harvest outages, backhaul bottlenecks, and sudden traffic surges. Stability evidence requires long-horizon queue behavior and offered-load context; a finite flat-looking curve alone is not a proof.

## 15. Verification and expected deliverables

Before learning experiments, verify work conservation across split/transport stages, CPU and radio capacity limits, stage causality, persistent arrivals, completion timestamps, interrupted uploads, harvest timing/overflow, energy ownership, virtual queue updates, and trajectory feasibility. Use small analytically checkable scenarios, including one vehicle/one server and complete-cloud/complete-UAV split endpoints.

Expected deliverables are a task-persistent VEC simulator, physically feasible hybrid queue-aware R-MADDPG, reproducible meta-training/adaptation scripts, fair baselines and ablations, evaluation traces/configurations, and an IEEE report supported by those results.

The intended outcome is a controller that learns where work should go and where UAVs should move, adapts with limited new experience, and accounts for the downstream queues and future energy consequences of its choices. Whether it outperforms the baselines, remains stable under feasible loads, or offers a useful adaptation-speed advantage is what the experiments must establish.

## 16. Reproducible evaluation command

Phase 6 is implemented in `src/vec/evaluation` and exposed through
`scripts/evaluate_vec.py`. The evaluator materializes each exogenous task and
harvest trace once, hashes it, and reuses it across methods. It saves the full
environment configuration, scenario split, seed, code revision/dirty flag,
runtime versions, per-slot trace, per-task censored outcome, aggregate metrics,
95% seed-level confidence intervals, paired differences, figures, and SHA-256
artifact hashes.

A small non-claim smoke run is:

```powershell
python scripts/evaluate_vec.py `
  --output output/evaluation/smoke `
  --split test --seeds 900,901 `
  --stresses low_load,traffic_surge `
  --horizon 50 --drain 50 `
  --methods local_only,queue_energy_greedy
```

Register a learned MADDPG- or MAPPO-family checkpoint with
`--checkpoint METHOD=PATH`; the evaluator reads the checkpoint's algorithm
metadata and selects the matching controller. Register any additional
common-interface controller with
`--controller-factory METHOD=MODULE:CALLABLE`. Use
`--require-complete-roster` for final paper runs; it aborts if any required
comparison is absent. Without that flag, missing methods are still written to
`evaluation_status.json`, and the bundle is explicitly marked incomplete.

For Bologna/SUMO evaluation, export SUMO floating-car data and convert a
complete fixed-population segment with `scripts/convert_sumo_fcd.py`. Set the
resulting CSV as `mobility_trace` in each scenario manifest and list the three
selected vehicle IDs in `mobility_vehicle_ids`. The loader rejects missing
frames and the manifest rejects reuse of the same mobility trace across train,
validation, and test splits.

The generated `metric_protocol.json` fixes the statistical definitions used in
the tables. In particular, deadline success includes censored tasks in its
denominator, latency percentiles cover completed tasks only and are always
accompanied by the censor count, backlog slopes exclude drain slots, and paired
comparisons reject mismatched scenario keys.

## 17. Reproducible learned-policy training

Use `scripts/train_vec.py` for the new VEC learner; the legacy root `train.py`
continues to serve the earlier DMJO experiment. The three supported modes are
`maddpg` (one declared base stress), `domain_randomized` (one shared policy
across varied training scenarios), and `reptile` (task-local adaptation plus a
first-order meta update). For example:

```powershell
python scripts/train_vec.py `
  --output output/training/domain-randomized-seed0 `
  --mode domain_randomized --epochs 50 --seed 0
```

Add `--no-queue-weight` for the R-MADDPG physical-queue-term ablation. This
changes only the learned objective; physical batteries, trajectory limits, and
the common feasibility controller remain active. `--resume` restores model,
target, optimizer, replay, trainer RNG, epoch, and early-stopping state from an
interrupted run. `--smoke` deliberately reduces the scenario and network size
and must never be cited as final training.

Each output directory contains the exact training configuration and scenario
manifest, repository revision/dirty state, append-only epoch history, resume
state, the best safety-eligible validation checkpoint (when one exists), the
last debugging checkpoint, and a wall-clock summary. An unsafe run does not
silently promote `last.pt` to `best.pt`.
The bundle also includes SHA-256 hashes and runtime/library versions.
Training reads only the manifest's `train` and `validation` splits. The sealed
`test` split is used later by `scripts/evaluate_vec.py`, never for checkpoint
selection.

The independent on-policy comparison uses `scripts/train_vec_mappo.py`:

```powershell
python scripts/train_vec_mappo.py `
  --output output/training/hybrid-mappo-seed0 `
  --epochs 50 --seed 0
```

It uses role-shared masked hybrid actors, a centralized state-value critic,
GAE, clipped PPO updates, entropy regularization, KL early stopping, and an
auxiliary proposal-to-execution repair loss. Its checkpoint carries an
algorithm tag, so `scripts/evaluate_vec.py --checkpoint
hybrid_mappo=PATH_TO_CHECKPOINT` selects the MAPPO controller automatically.
The root `src/mappo_agent.py` remains the legacy diffusion-MAPPO implementation
and is not used as evidence for the queue-aware VEC comparison.

## 18. Complete multi-seed study

Use `scripts/run_vec_study.py` to execute the final experiment matrix. It
creates one immutable scenario manifest, trains ordinary MADDPG,
domain-randomized MADDPG, R-MADDPG, the queue-weight ablation, and MAPPO for
every learner seed, and then evaluates each seed's checkpoint set together
with the four required non-learning baselines.

Generate and inspect the default five-seed plan without starting training:

```powershell
python scripts/run_vec_study.py `
  --output output/studies/research-v1 `
  --dry-run
```

Run or resume that frozen study with the identical command after removing
`--dry-run`. Completed jobs are skipped and interrupted jobs with saved state
resume automatically. A complete integration check is available through:

```powershell
python scripts/run_vec_study.py `
  --output output/studies/smoke-v1 `
  --learner-seeds 0 --epochs 1 --scenarios-per-epoch 1 `
  --smoke --no-plots
```

The final confidence intervals and paired comparisons use the independently
trained learner seed as the statistical unit. Test scenarios are first
averaged within each learner seed, preventing time slots or correlated test
scenarios from being presented as independent training replications. The
study summary refuses research-ready status for smoke runs, incomplete method
rosters, fewer than five learner seeds, a run below the frozen minimum
research scale, or any recorded physical, battery, or separation violation.
Protocol readiness still does not prove convergence or algorithm superiority;
learning curves and effect estimates must be inspected.
