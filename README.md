# Sim-to-Sim Transfer through Domain Randomization

Reinforcement Learning project studying how **Uniform Domain Randomization (UDR)** affects the robustness and generalization of locomotion policies under dynamics mismatch.

The project evaluates **sim-to-sim transfer** in a custom **MuJoCo Hopper** environment, comparing policies trained under nominal dynamics with policies exposed to randomized physical parameters.

Beyond standard episodic return, the project introduces a deeper robustness analysis based on:

- Failure dynamics
- Time-to-failure
- Near-fall events
- Torso stability
- Policy network capacity
- Latent representation analysis using PCA and t-SNE

The results show that **moderate domain randomization significantly improves transfer performance and stability**, while excessive randomization can negatively affect both the learned representations and the resulting control policy.

---

## Overview

Reinforcement Learning policies trained in simulation often suffer a significant performance drop when deployed in environments whose dynamics differ from those used during training.

Even relatively small changes in physical parameters such as mass or inertia can cause a locomotion policy to become unstable.

This project investigates **Domain Randomization (DR)** as a strategy to improve robustness against these dynamics variations.

The main objective is not only to determine whether domain randomization improves the final reward, but also to understand **why certain randomization strategies produce more robust policies**.

The study focuses on five main questions:

1. Which Hopper body components are most useful to randomize?
2. How does the magnitude of domain randomization affect sim-to-sim transfer?
3. Does increasing policy network capacity improve robustness?
4. How do different policies fail under dynamics mismatch?
5. How does domain randomization affect the internal representations learned by the policy?

---

## Sim-to-Sim Transfer Setup

Two versions of a custom MuJoCo Hopper environment are considered:

### Source Domain

The source environment corresponds to the nominal Hopper model.

All physical parameters remain fixed at their default values during evaluation.

This environment is also used for policy training.

### Target Domain

The target environment introduces a dynamics shift by modifying the masses of selected Hopper components.

The following body parts are considered:

- Torso
- Thigh
- Leg
- Foot

These modifications change the inertial and contact properties of the system, creating a controlled **sim-to-sim transfer problem**.

### Source vs Target Environment

The following figure illustrates the difference between the nominal source environment and the modified target domain used for transfer evaluation.

<p align="center">
  <img src="results/Comparison_SourceTarget.png" width="750">
</p>

The policy is trained exclusively in the source environment and then evaluated in both the source and target domains to measure its ability to generalize under modified dynamics.

The general experimental pipeline is:

```text
        Source Environment
               │
               ▼
      Reinforcement Learning
            Training
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   No Domain        Uniform Domain
 Randomization      Randomization
       │                │
       └───────┬────────┘
               │
               ▼
         Trained Policy
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
 Source Evaluation   Target Evaluation
                         │
                         ▼
                  Transfer Analysis
```

---

## Uniform Domain Randomization

Domain Randomization is applied during training by modifying physical parameters at the beginning of each episode.

For a body component with nominal mass \(m_0\), its randomized mass is sampled from:

\[
m \sim U((1-\delta)m_0,\,(1+\delta)m_0)
\]

where \(\delta\) defines the randomization magnitude.

Different experiments vary:

- The Hopper components being randomized
- The randomization strength
- The policy network architecture

The following randomization strengths were evaluated:

```text
δ = 0.05
δ = 0.10
δ = 0.20
δ = 0.30
δ = 0.40
```

depending on the experiment.

---

## Reinforcement Learning

Policies are trained using **Proximal Policy Optimization (PPO)** with an actor-critic architecture.

Both the policy and value functions are represented using multilayer perceptrons with ReLU activations.

Unless explicitly modified in an experiment, all PPO hyperparameters remain fixed to ensure a fair comparison between different domain randomization strategies.

The main policy architectures evaluated were:

```text
[64, 64]

[128, 128]

[256, 256]
```

---

# Experiments

## Experiment 1 — Which Body Parts Should Be Randomized?

The first experiment studies how randomizing different Hopper components affects transfer to the target domain.

Each component is randomized independently using:

```text
δ = 0.20
```

A configuration in which all components are randomized simultaneously is also evaluated.

### Results

| Randomized parts | Source Return | Target Return |
|---|---:|---:|
| None — Baseline | 1569 ± 6 | 835 ± 18 |
| Thigh only | 1381 ± 184 | 1039 ± 29 |
| Leg only | 1016 ± 117 | 708 ± 15 |
| Foot only | 1523 ± 6 | 1266 ± 252 |
| All body parts | **1543 ± 11** | **1320 ± 109** |

The results show that not every form of randomization is equally useful.

Randomizing the **foot** produces a large increase in target performance while largely preserving source performance.

Randomizing all body components provides the best overall transfer result, increasing the target return from:

```text
835 ± 18
```

for the baseline policy to:

```text
1320 ± 109
```

with full-body domain randomization.

The experiment suggests that parameters related to **contact dynamics**, particularly the foot, have an important influence on policy robustness.

---

## Experiment 2 — Randomization Strength

After identifying foot-only and full-body randomization as the most promising configurations, the second experiment studies the influence of the randomization magnitude.

### Foot Randomization

| δ | Source Return | Target Return |
|---:|---:|---:|
| 0.05 | 1323 ± 138 | 1004 ± 39 |
| 0.10 | 1533 ± 4 | 1155 ± 44 |
| **0.20** | **1523 ± 6** | **1266 ± 252** |
| 0.40 | 1678 ± 15 | 1105 ± 71 |

### Full-Body Randomization

| δ | Source Return | Target Return |
|---:|---:|---:|
| 0.05 | 1627 ± 10 | 807 ± 90 |
| 0.10 | 1099 ± 71 | 884 ± 19 |
| **0.20** | **1543 ± 11** | **1320 ± 109** |
| 0.30 | 1002 ± 32 | 695 ± 2 |
| 0.40 | 1208 ± 22 | 817 ± 27 |

A clear trade-off appears.

### Insufficient randomization

When the randomization range is too small, the training distribution does not sufficiently cover the target dynamics.

### Excessive randomization

Very large randomization ranges make the optimization problem more difficult and can reduce both source and target performance.

### Best configuration

In both experiments:

```text
δ = 0.20
```

provides the best balance between robustness and learning stability.

This indicates that **more domain randomization is not necessarily better**.

---

## Experiment 3 — Policy Network Capacity

The third experiment investigates whether larger policy networks improve robustness under domain randomization.

Full-body randomization with:

```text
δ = 0.20
```

is used while varying the MLP architecture.

Each architecture is evaluated across multiple random seeds.

| Architecture | Source Return | Target Return |
|---|---:|---:|
| [64, 64] | 1551 ± 81 | 1048 ± 238 |
| [128, 128] | 1476 ± 234 | 1234 ± 453 |
| **[256, 256]** | **1603 ± 163** | **1294 ± 350** |

Increasing network capacity generally improves average target performance.

However, larger architectures also exhibit greater variability between training seeds.

The results therefore show a trade-off between:

```text
Network capacity
      ↑
Representation capability
      ↑
Average robustness
```

and:

```text
Network capacity
      ↑
Training variance
      ↑
Reduced consistency
```

---

# Stability and Failure Analysis

Average episodic return alone does not fully explain whether a locomotion policy is robust.

Two policies may achieve similar returns while displaying very different levels of stability.

For this reason, additional physical metrics are recorded during evaluation:

- Episode length
- Time-to-failure
- Minimum torso height
- Torso angle standard deviation
- Near-fall events

A **near-fall** occurs when the torso height drops below a predefined safety threshold without immediately terminating the episode.

Three representative policies are compared:

```text
No Domain Randomization

Moderate DR
δ = 0.20

Excessive DR
δ = 0.40
```

### Target-domain stability results

| Policy | Timeout | Time-to-Fail | Min. Torso Height | Torso Angle Std. | Near-Falls |
|---|---:|---:|---:|---:|---:|
| No DR | 0% | 366 | 0.687 | 0.035 | 3.3 |
| **DR δ = 0.20** | **100%** | **3000** | **0.978** | **0.020** | **0.0** |
| DR δ = 0.40 | 0% | 464 | 0.686 | 0.062 | 3.0 |

The differences are significant.

### Baseline policy

The policy trained without domain randomization progressively becomes unstable in the target environment.

It exhibits:

- Short time-to-failure
- Multiple near-falls
- Low minimum torso height
- Small stability margin

### Moderate Domain Randomization

The policy trained with:

```text
δ = 0.20
```

shows fundamentally different behavior.

It:

- Reaches the maximum episode length
- Completes all evaluation episodes without failure
- Maintains a substantially higher torso height
- Produces smoother angular motion
- Experiences no near-fall events

### Excessive Domain Randomization

The policy trained with:

```text
δ = 0.40
```

performs slightly better than the baseline in terms of time-to-failure, but exhibits much larger torso-angle oscillations.

This suggests that excessive randomization introduces variability into the learned control strategy and ultimately reduces locomotion stability.

---

# Latent Representation Analysis

To better understand why moderate domain randomization improves robustness, the internal representations of the policy network are analyzed.

During evaluation, activations are extracted from the **penultimate layer of the policy network**.

These latent vectors represent the internal encoding used by the policy before selecting an action.

Representations are collected in both:

```text
Source Environment
        +
Target Environment
```

and projected into two dimensions using:

- **Principal Component Analysis (PCA)**
- **t-distributed Stochastic Neighbor Embedding (t-SNE)**

PCA is used to analyze the global structure of the latent space, while t-SNE is used to inspect local neighborhood relationships.

---

## No Domain Randomization

For the baseline policy, source and target states occupy noticeably different regions of the latent space.

This indicates that the policy internally represents similar physical states differently depending on the environment in which they occur.

Such **domain-dependent representations** are consistent with the poor transfer performance and early failures observed in the target environment.

---

## Moderate Domain Randomization

With:

```text
δ = 0.20
```

source and target representations become substantially better aligned.

In the PCA projection, both domains largely overlap.

The t-SNE representation also shows source and target samples mixed inside common local clusters.

This suggests that the policy has learned more **domain-invariant features**.

Instead of strongly depending on the exact simulation dynamics, the policy learns representations that capture state information useful across both domains.

This representational alignment is associated with:

- Higher target return
- Larger stability margin
- Lower torso oscillation
- No near-fall events

---

## Excessive Domain Randomization

With:

```text
δ = 0.40
```

source and target representations may appear mixed, but the latent space becomes more dispersed and less structured.

The policy therefore gains some invariance at the cost of losing meaningful representation structure.

This behavior is consistent with the increased oscillations and reduced stability observed during locomotion.

---

# Main Findings

The experiments reveal several important properties of Domain Randomization for sim-to-sim transfer.

### 1. Domain Randomization improves transfer

The no-DR policy obtains:

```text
Target Return = 835 ± 18
```

while full-body DR with:

```text
δ = 0.20
```

achieves:

```text
Target Return = 1320 ± 109
```

while maintaining strong source-domain performance.

---

### 2. Randomizing the right parameters matters

Randomizing all physical parameters blindly is not the only effective strategy.

Foot-only randomization already provides a large robustness improvement:

```text
Target Return

No DR:
835 ± 18

Foot DR:
1266 ± 252
```

indicating the particular importance of parameters related to contact dynamics.

---

### 3. Moderate randomization is better than maximum randomization

The best-performing configuration uses:

```text
δ = 0.20
```

Increasing the range to:

```text
δ = 0.40
```

does not improve robustness and can substantially reduce stability.

---

### 4. Robustness is not fully described by reward

The stability analysis shows that domain-randomized policies are not simply obtaining higher return.

The best policy also:

- Avoids near-falls
- Maintains a higher torso position
- Produces smoother motion
- Survives the complete evaluation horizon

---

### 5. Robust policies learn more domain-invariant representations

The latent representation analysis shows a strong relationship between successful transfer and internal representation alignment.

Moderate DR encourages the policy to map source and target states into similar regions of the latent space.

This provides a possible explanation for the improved transfer robustness.

---

# Repository Structure

```text
.
├── train.py
│   └── PPO policy training and Domain Randomization
│
├── check_udr.py
│   └── Validation of the Uniform Domain Randomization implementation
│
├── test_random_policy.py
│   └── Policy evaluation under randomized dynamics
│
├── analyze_failures.py
│   └── Failure and stability analysis
│
├── analyze_representations.py
│   └── Extraction and visualization of policy latent representations
│
├── report_project_2025.ipynb
│   └── Experimental analysis notebook
│
├── results/
│   ├── Comparison_SourceTarget.png
│   ├── ExperimentosAblacion.txt
│   ├── ExperimentosEstructuras.txt
│   ├── ExperimentosFallos.txt
│   └── ExperimentosSweep.txt
│
└── Carlo_Squarcia_Paper.pdf
    └── Complete project report
```
---

# Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/ercarlo/sim-to-sim-domain-randomization.git
cd sim-to-sim-domain-randomization
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train a policy

```bash
python train.py
```

## 5. Evaluate the randomization strategy

```bash
python test_random_policy.py
```

## 6. Analyze failure behavior

```bash
python analyze_failures.py
```

## 7. Analyze latent representations

```bash
python analyze_representations.py
```

> Exact parameters and experimental configurations can be modified directly in the corresponding training and evaluation scripts.

---

# Evaluation Methodology

Each trained policy is evaluated in two scenarios:

```text
Source → Source
```

Training and evaluation are both performed using the nominal dynamics.

and:

```text
Source → Target
```

The policy is trained in the source environment and evaluated under modified target dynamics.

Performance is measured using:

```text
Mean Episodic Return ± Standard Deviation
```

over multiple evaluation episodes and random seeds.

This allows the transfer performance to be compared against the nominal policy performance.

---

# Technologies

The project combines:

- **Python**
- **Reinforcement Learning**
- **Proximal Policy Optimization (PPO)**
- **MuJoCo**
- **Domain Randomization**
- **Actor-Critic Neural Networks**
- **Multilayer Perceptrons**
- **PCA**
- **t-SNE**
- **Sim-to-Sim Transfer**
- **Robotic Locomotion**
- **Representation Analysis**

---

# Project Contributions

The main contributions of this project are:

- Implementation of **Uniform Domain Randomization** for physical parameter variation in a custom Hopper environment.
- Analysis of the contribution of individual body components to robustness.
- Experimental study of the optimal randomization magnitude.
- Analysis of the relationship between policy network capacity and transfer performance.
- Introduction of stability-oriented metrics beyond episodic return.
- Detailed analysis of failure dynamics and near-fall behavior.
- Analysis of policy latent representations using PCA and t-SNE.
- Study of the relationship between representation invariance and sim-to-sim robustness.

---

# Conclusion

This project demonstrates that effective sim-to-sim transfer depends not only on applying Domain Randomization, but on **choosing an appropriate randomization strategy**.

Moderate randomization provides the strongest results, simultaneously improving:

```text
Target-domain performance
          +
Locomotion stability
          +
Failure resistance
          +
Representation invariance
```

Excessive randomization, on the other hand, can reduce the structure of the learned latent space and introduce instability into the control policy.

The results therefore suggest that robust transfer emerges from a balance between **domain invariance and representation structure**, rather than from maximizing the amount of randomization applied during training.

---

# Project Report

A detailed description of the methodology, experiments and results is available in the complete project paper:

[**Studying Domain Randomization Strategies for Sim-to-Sim Transfer in Reinforcement Learning**](./Carlo_Squarcia_Paper.pdf)

---

## Author

**Carlo Squarcia Mateo**

Project focused on Reinforcement Learning, Domain Randomization and robust robotic control.

---

## References

This project builds upon fundamental work in Reinforcement Learning and Domain Randomization, including:

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov,  
   *Proximal Policy Optimization Algorithms*, 2017.

2. J. Tobin et al.,  
   *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS, 2017.

3. X. B. Peng, M. Andrychowicz, W. Zaremba and P. Abbeel,  
   *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*, ICRA, 2018.

4. A. Rajeswaran et al.,  
   *EPOpt: Learning Robust Neural Network Policies Using Model Ensembles*, ICLR, 2017.

5. A. Zhang, R. McAllister and S. Levine,  
   *Learning Invariant Representations for Reinforcement Learning*, ICLR, 2018.
