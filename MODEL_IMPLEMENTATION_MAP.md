# System-model to implementation traceability

This file connects the active equations in `report/IEEEtran/main.tex` to the
implementation and its verification. It is intended for research review and
presentation questions. The report remains the mathematical definition; the
code paths below show where each definition is executed.

## Slot timing contract

At the start of a slot, the controller observes the current queues, positions,
batteries, and destination masks. Work already eligible at that boundary may
be dispatched or served. Upload, staging, backhaul, and compute departures are
recorded explicitly. Completed transfers enter their downstream queue for the
next causal stage; new external arrivals are committed at the next boundary.
Motion and energy are then committed, end-of-slot harvesting is applied, and
deadline crossings are recorded. This order is implemented in
`VecQueueEnv.step` and is tested with deterministic single-slot and multi-slot
traces.

## Equation and model map

| Report model | Implemented behavior | Main code | Verification |
|---|---|---|---|
| Vehicle mobility and distance geometry, Eqs. (1)-(2) | Road-segment motion or complete recorded frames; 2D V2R and 3D V2U distances | `src/vec/mobility.py`, `src/vec/env.py` | `tests/test_vec_radio_mobility.py`, `tests/test_vec_phase6.py` |
| Task arrivals and persistent task record, Eqs. (3)-(4) | Identity, input bits, cycles, relative deadline, split state, completion time, and deadline-miss state persist until completion/censoring | `src/vec/tasks.py`, `src/vec/env.py` | `tests/test_vec_phase1.py`, `tests/test_vec_cloud_split.py` |
| Unique dispatch and UAV/cloud split, Eqs. (5)-(6) | A held ingress task may later select one feasible destination; a partitionable UAV task retains one fixed local/cloud fraction and conserves equivalent cycles | `src/vec/env.py`, `src/vec/control.py` | `tests/test_vec_cloud_split.py`, `tests/test_vec_phase1.py` |
| End-to-end response time, Eq. (7) | Completion timestamp minus arrival timestamp; a split task completes only after all nonempty branches; unfinished work is censored | `src/vec/tasks.py`, `src/vec/evaluation/metrics.py` | `tests/test_vec_phase1.py`, `tests/test_vec_phase6.py` |
| Terrestrial/aerial gain, LoS probability, Shannon rate, and OFDMA limits, Eqs. (8)-(12) | Coverage-aware V2R/V2U link gains, fixed transmit power, orthogonal bandwidth sharing, rate and bit budgets | `src/vec/radio.py`, `src/vec/env.py` | `tests/test_vec_radio_mobility.py` |
| Persistent queue evolution and flow conservation, Eqs. (13)-(14) | Ingress, upload, receive/staging, processing, UAV backhaul, and cloud queues retain residual work; internal transfers conserve equivalent cycles | `src/vec/queues.py`, `src/vec/env.py`, `src/vec/lyapunov.py` | `tests/test_vec_phase1.py`, `tests/test_vec_phase3.py`, `tests/test_vec_cloud_split.py` |
| CPU capacity and weighted processor sharing, Eqs. (15)-(16) | Each node has a finite per-slot cycle budget; nonnegative normalized task weights share that budget without serving more than eligible work | `src/vec/queues.py`, `src/vec/env.py` | `tests/test_vec_phase1.py`, `tests/test_vec_energy_safety.py` |
| Strong-stability target, Eq. (17) | Not asserted from finite experiments; evaluation reports final/peak backlog and offered-horizon growth by stage, including overload | `src/vec/evaluation/metrics.py`, `src/vec/evaluation/scenarios.py` | `tests/test_vec_phase6.py` |
| 3D UAV kinematics, bounds, and continuous-segment separation, Eqs. (18)-(20) | Velocity, acceleration, altitude, map, climb/descent, and closest-approach constraints are checked jointly; infeasible proposals are repaired or rejected | `src/vec/safety.py`, `src/vec/env.py` | `tests/test_vec_energy_safety.py`, `tests/test_vec_env_energy_safety.py` |
| Rotary-wing propulsion and service-energy components, Eqs. (21)-(23) | Propulsion, computation, reception, backhaul, control, vehicle, RSU, and cloud energy are realized once and retained as separate ledger terms | `src/vec/energy.py`, `src/vec/env.py` | `tests/test_vec_energy_safety.py`, `tests/test_vec_phase6.py` |
| Stochastic harvesting and battery update, Eqs. (24)-(26) | Nonnegative bounded harvest is unknown to the action and becomes available after expenditure; overflow is measured and charge is capacity-clipped | `src/vec/energy.py`, `src/vec/env.py`, `src/vec/evaluation/scenarios.py` | `tests/test_vec_energy_safety.py`, `tests/test_vec_env_energy_safety.py` |
| Verified return/landing reserve, Eq. (27) | Normal service is allowed only when post-action charge covers a complete feasible landing plan; forced-return UAVs reject new work and land without free hovering | `src/vec/return_plan.py`, `src/vec/env.py` | `tests/test_vec_env_energy_safety.py`, `tests/test_vec_phase6.py` |
| Instantaneous cost and long-term objective, Eqs. (28)-(29) | Dense unfinished-task occupancy, whole-system energy, and first deadline crossings form the immediate cost; hard feasibility remains outside the reward trade-off | `src/vec/lyapunov.py`, `src/vec/env.py` | `tests/test_vec_phase3.py` |
| Virtual energy deficit and drift-plus-penalty, Eqs. (30)-(32) | Normalized physical queues and net-energy virtual queues produce the recorded drift bound/objective without replacing physical battery checks | `src/vec/lyapunov.py` | `tests/test_vec_phase3.py` |
| Hybrid MADDPG critic/actor updates, Eqs. (33)-(34) | Role actors produce masked categorical and bounded continuous decisions; centralized critics use joint state/actions; replay stores executed repaired actions | `src/vec/learning/networks.py`, `src/vec/learning/agent.py`, `src/vec/learning/adapter.py` | `tests/test_vec_phase4.py`, `tests/test_vec_phase5.py` |
| Reptile outer update, Eq. (35) | Each sampled environment distribution receives isolated replay/optimizer state, inner adaptation, and a first-order actor-and-critic outer update | `src/vec/learning/meta.py`, `src/vec/training/trainer.py` | `tests/test_vec_phase5.py`, `tests/test_vec_training.py` |
| Independent MAPPO comparator | Role-shared stochastic hybrid actors, centralized value learning, GAE, clipped PPO, entropy/KL diagnostics, and executed-action repair regularization | `src/vec/learning/mappo.py`, `src/vec/training/mappo_trainer.py` | `tests/test_vec_mappo.py` |
| Evaluation statistics and artifacts | Test scenarios are paired across methods; learner seed is the uncertainty unit; all pairwise effects, traces, checkpoints, versions, and hashes are retained | `src/vec/evaluation`, `src/vec/experiments.py` | `tests/test_vec_phase6.py`, `tests/test_vec_study.py` |

## Deliberate scope limits

- Returned result packets are assumed negligible; a substantial downlink would
  require another persistent radio queue.
- The frozen `research-v1` study uses procedurally generated road mobility,
  task arrivals, and harvesting. Recorded SUMO mobility is supported but is not
  part of that result bundle.
- A finite horizon does not prove queue stability. Near-capacity and overload
  traces are reported with backlog growth and censoring instead.
- The optional laser-transfer expression in the report is not used by the
  stochastic-harvest benchmark and must not be presented as a calibrated laser
  experiment.
- Feasibility enforcement demonstrates that executed actions satisfy the
  implemented constraints; it does not prove global policy optimality.
