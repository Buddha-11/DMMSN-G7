# Queue-Aware Meta-Reinforcement Learning for Multi-UAV Assisted Vehicular Edge Computing

## 1. Final Problem Statement

Modern vehicular applications generate computation-intensive and latency-sensitive tasks while vehicles continuously move through the network. UAVs equipped with edge servers can dynamically move toward high-demand areas and provide additional computing capacity. However, an effective system must simultaneously handle four coupled challenges: changing vehicle mobility and network conditions, limited and uneven computing resources, growing task queues, and limited UAV energy.

Our problem is therefore:

> To design an adaptive multi-UAV-assisted vehicular edge-cloud computing framework that jointly decides task offloading, computing-resource allocation, and 3-D UAV trajectories, while maintaining long-term task-queue stability and satisfying UAV energy constraints, including energy harvesting, under dynamically changing vehicle mobility and workload conditions.

The objective is to reduce task execution latency and energy consumption while keeping computational queues stable and UAVs operational over the long term.

---

## 2. What issues are we fixing and how?

| Existing Problem | What happens | Concept we take | How it fixes it |
| :--- | :--- | :--- | :--- |
| **1. Vehicles are constantly moving** | The best UAV position and offloading decision changes with time | 3-D UAV path planning (Q-TOP/Paper 2) | UAVs dynamically adjust their trajectories according to changing vehicle/task demand |
| **2. Network conditions change** | A policy that worked in one traffic/network situation may perform poorly in another | Meta-learning (Paper 1) | Agents can rapidly adapt to new environments instead of requiring complete retraining |
| **3. Multiple UAVs need to coordinate** | Independent decisions can cause inefficient resource usage or conflicts | MADDPG (Paper 1) | Multiple UAVs/edge nodes learn cooperative decisions |
| **4. Some UAV/RSU queues become overloaded** | Tasks accumulate, increasing waiting and execution latency | Queue-aware decision-making + Lyapunov (Q-TOP/Paper 2) | Queue backlog becomes part of the optimization; Lyapunov handles long-term queue stability |
| **5. UAVs have limited battery** | Flying, communication and computing consume energy | Energy model/constraint (Q-TOP/Paper 2) | UAV decisions account for energy consumption rather than optimizing latency alone |
| **6. UAVs can replenish energy** | Battery limitation can restrict continuous operation | Energy harvesting (Q-TOP/Paper 2) | Harvested energy is incorporated into UAV energy management |
| **7. Edge resources may not be enough** | UAV/RSU can become overloaded | Edge-cloud architecture (Paper 1) | Remaining tasks can be forwarded to cloud resources when edge capacity is insufficient |
| **8. Need to balance latency and energy** | Minimizing only latency may consume excessive energy, and vice versa | Energy + latency reward (Paper 1 & 2) | Learning objective considers both latency and energy |
| **9. Decisions are coupled** | Where the UAV flies affects communication; communication affects offloading; offloading affects queues and energy | Combination of both papers | A unified framework jointly considers these decisions |

---

## 3. The actual flow of our solution

Imagine 100 cars on a highway generating tasks.

### Step 1 — Cars generate tasks
Each vehicle generates something like: *"I have a computation task of size X, requiring Y CPU cycles, and it needs to finish within Z time."*

The system observes: Vehicle locations, mobility, task arrivals/sizes/deadlines, UAV positions/energy, UAV/RSU queue lengths, and available computing resources.

### Step 2 — Meta-MADDPG makes decisions (The AI Brain)
It asks: *"Given the current situation, what should the UAVs and edge nodes do?"*
It learns decisions such as: Which task should be offloaded? Which UAV should serve it? Should it go to an RSU, remain local, or use cloud resources? How should computing resources be allocated? Where should the UAV move?

The meta-learning component ensures that when the environment changes (e.g., test environment differs from training), the policy can rapidly adapt rather than degrade.

### Step 3 — UAV path planning
If the AI sees a cluster of cars generating high demand, the Meta-MADDPG decides the movement action, and the 3-D path planning determines the UAV's trajectory over time to move toward that region, coupling trajectories with task allocation.

### Step 4 — Queue awareness
If UAV 1 has a queue of 90 tasks and UAV 2 has 10, simply choosing the closest UAV (UAV 1) is a bad decision. The system recognizes "UAV 1 is already overloaded" and redirects work toward UAV 2, an RSU, local execution, or the cloud.

### Step 5 — Lyapunov optimization (The Long-Term Discipline Layer)
The problem isn't just *"What is the best decision right now?"* It is *"Can I keep making good decisions without allowing queues to grow indefinitely?"*
Lyapunov optimization converts the long-term stochastic queue/energy constrained problem into manageable per-time-slot decisions, wrapping a long-term discipline layer around the RL decisions.

### Step 6 — Energy management
Sending a low-battery UAV far away increases propulsion energy risk. The system considers current energy, movement/communication/computing energy costs, and replenished energy from harvesting to keep UAVs operational.

### Step 7 — Edge or Cloud Hierarchy
Tasks are processed through an available hierarchy: Vehicle $\rightarrow$ RSU $\rightarrow$ UAV $\rightarrow$ Cloud (if edge capacity is insufficient).

---

## 4. The 5 Major Decisions Solved

Our framework jointly optimizes:
1. **Task Offloading**: Where should the task execute? (Vehicle / UAV / RSU / Cloud)
2. **Resource Allocation**: How much computing resource should each task receive?
3. **UAV Trajectory**: Where should each UAV fly?
4. **Queue Management**: How do we prevent task queues from becoming unstable?
5. **Energy Management**: How do we keep UAVs operational while accounting for consumption and harvested energy?

*(Meta-MADDPG is the learning mechanism coordinating these decisions under Lyapunov constraints).*

---

## 5. The Novelty (How to explain it)

> *"Existing approaches separately address adaptive computation offloading and queue-aware UAV path planning. We aim to integrate adaptive meta-reinforcement learning with Lyapunov-based queue and energy management to jointly optimize task offloading, resource allocation, and 3-D UAV trajectory decisions in a dynamic, energy-harvesting edge-cloud vehicular environment."*

### System Architecture Integration
```text
DYNAMIC VEHICULAR ENVIRONMENT
                              │
                ┌─────────────┴─────────────┐
                │                           │
        Vehicle Mobility              Task Arrivals
                │                           │
                └─────────────┬─────────────┘
                              ↓
                    ┌─────────────────┐
                    │  META-MADDPG    │
                    │  Adaptive Brain │
                    └────────┬────────┘
                             ↓
             ┌───────────────┼───────────────┐
             ↓               ↓               ↓
        Offloading      Resource         UAV 3-D
         Decision       Allocation       Movement
             │               │               │
             └───────────────┼───────────────┘
                             ↓
                  ┌────────────────────┐
                  │ LYAPUNOV MANAGEMENT│
                  └─────────┬──────────┘
                            ↓
                 ┌──────────┴──────────┐
                 ↓                     ↓
          Queue Stability        Energy Constraint
                                       │
                                Energy Harvesting
                                       ↓
                             Sustainable UAV MEC
                                       │
                                       ↓
                              Lower Latency +
                              Lower Energy +
                              Stable Queues
```

---

## 6. Codebase Analysis (`src/`) & Implementation Plan

The current code in `src/` implements a **Diffusion-MAPPO (Multi-Agent PPO with Diffusion)** algorithm for a multi-UAV network. 

**Codebase Gap Analysis:**
The existing code (`env.py`, `mappo_agent.py`) is an excellent foundation for multi-UAV RL, currently geared towards secure NOMA uplink collection and cooperative caching. It lacks the explicit **Lyapunov queue stability**, **energy harvesting mechanisms**, and the **Reptile meta-learning loop** required by our merged problem statement.

### Implementation Plan (Extending `src/`)

**Phase 1: Environment Upgrades (`env.py`)**
1.  **Queue Dynamics:** Introduce explicit task buffers (queues) for `IoTDevice`, `UAV`, and a new `RSU` entity. Implement queue update equations.
2.  **Energy Harvesting:** Add an energy harvesting model to the `UAV` class, simulating random energy arrivals. Implement virtual queues to track energy deficits.
3.  **Reward Function Formulation:** Modify the step reward to return the negative **Lyapunov drift-plus-penalty**. 
    *   $Reward = - (V \cdot \text{SystemCost} + \sum (\text{Queue Backlogs} \times \text{Arrivals/Processing}))$

**Phase 2: Algorithm Implementation (R-MADDPG)**
1.  **Inner Module (`rmaddpg_agent.py`):** Create a standard MADDPG agent capable of outputting continuous actions for 3D trajectory $\Delta x, \Delta y, \Delta z$ and offloading ratios.
2.  **Outer Module (Reptile Meta-Learning):** Create a new training script (`train_meta.py`).
    *   Use the existing `DomainRandomiser` to sample a batch of diverse tasks/environments.
    *   For each task, clone the meta-weights, run the inner MADDPG agent, and calculate the task-specific gradients.
    *   Apply the Reptile update rule: $\theta_{meta} \leftarrow \theta_{meta} + \epsilon (\theta_{task} - \theta_{meta})$.

**Phase 3: Action and State Space Mapping**
*   Update `get_local_obs()` and `get_global_state()` to include current task queue backlogs, UAV battery levels, and energy deficit virtual queues so the agent can learn queue-aware policies.
