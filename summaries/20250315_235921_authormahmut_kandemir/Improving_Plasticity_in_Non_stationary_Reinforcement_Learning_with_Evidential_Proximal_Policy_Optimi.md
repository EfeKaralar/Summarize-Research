# Improving Plasticity in Non-stationary Reinforcement Learning with Evidential Proximal Policy Optimization

**Authors:** Abdullah Akgül, Gulcin Baykal, Manuel Haußmann, Melih Kandemir

**Published:** 2025-03-03

**URL:** http://arxiv.org/abs/2503.01468v1

## Abstract

On-policy reinforcement learning algorithms use the most recently learned
policy to interact with the environment and update it using the latest gathered
trajectories, making them well-suited for adapting to non-stationary
environments where dynamics change over time. However, previous studies show
that they struggle to maintain plasticity$\unicode{x2013}$the ability of neural
networks to adjust their synaptic connections$\unicode{x2013}$with overfitting
identified as the primary cause. To address this, we present the first
application of evidential learning in an on-policy reinforcement learning
setting: $\textit{Evidential Proximal Policy Optimization (EPPO)}$. EPPO
incorporates all sources of error in the critic network's
approximation$\unicode{x2013}$i.e., the baseline function in advantage
calculation$\unicode{x2013}$by modeling the epistemic and aleatoric uncertainty
contributions to the approximation's total variance. We achieve this by using
an evidential neural network, which serves as a regularizer to prevent
overfitting. The resulting probabilistic interpretation of the advantage
function enables optimistic exploration, thus maintaining the plasticity.
Through experiments on non-stationary continuous control tasks, where the
environment dynamics change at regular intervals, we demonstrate that EPPO
outperforms state-of-the-art on-policy reinforcement learning variants in both
task-specific and overall return.

## Summary

### Comprehensive Summary of the Research Paper:

**Title:** Improving Plasticity in Non-stationary Reinforcement Learning with Evidential Proximal Policy Optimization  
**Authors:** Abdullah Akgül, Gulcin Baykal, Manuel Haußmann, Melih Kandemir  
**Published:** 2025-03-03  

---

### 1. Key Findings:
- **Evidential Proximal Policy Optimization (EPPO):** The paper introduces EPPO, the first application of evidential learning in on-policy reinforcement learning, which models epistemic and aleatoric uncertainty in the critic network to prevent overfitting and maintain plasticity.
- **Improved Plasticity:** EPPO outperforms state-of-the-art on-policy reinforcement learning algorithms in non-stationary environments by maintaining the ability of neural networks to adapt to changing dynamics.
- **Probabilistic Advantage Estimation:** EPPO introduces a probabilistic interpretation of the advantage function, enabling optimistic exploration and consistent performance improvements.
- **Experimental Validation:** EPPO demonstrates superior performance in non-stationary continuous control tasks, achieving higher task-specific and overall returns compared to existing methods.

---

### 2. Research Question/Problem:
The paper addresses the challenge of maintaining **plasticity** in on-policy reinforcement learning algorithms, particularly in **non-stationary environments** where the dynamics of the environment change over time. Existing methods, such as Proximal Policy Optimization (PPO), struggle with overfitting to past observations, leading to a loss of plasticity. The authors aim to solve this problem by introducing a novel approach that leverages evidential learning to model uncertainty and prevent overfitting.

---

### 3. Methodology:
- **Evidential Learning Framework:** EPPO incorporates evidential neural networks to model epistemic (model uncertainty) and aleatoric (data uncertainty) contributions to the total variance in the critic network's approximation of the value function.
- **Probabilistic Advantage Estimation:** The authors extend the Generalized Advantage Estimation (GAE) technique by introducing three probabilistic variants of the advantage function, enabling optimistic exploration.
- **Regularization:** The evidential neural network acts as a regularizer, preventing overfitting to past observations and maintaining plasticity.
- **Experimental Design:** The authors introduce two new experiment designs tailored for evaluating reinforcement learning algorithms in non-stationary continuous control tasks. They benchmark EPPO against state-of-the-art methods, including PPO variants and Proximal Feature Optimization (PFO).

---

### 4. Results:
- **Superior Performance:** EPPO consistently outperforms existing on-policy reinforcement learning algorithms in non-stationary environments, achieving higher task-specific and overall returns.
- **Plasticity Preservation:** EPPO maintains plasticity better than competing methods, as evidenced by its ability to adapt to changing dynamics over time.
- **Optimistic Exploration:** The probabilistic advantage estimators enable more effective exploration, contributing to improved performance in dynamic environments.

---

### 5. Implications:
- **Advancing Non-Stationary RL:** EPPO addresses a critical bottleneck in reinforcement learning by preserving plasticity in non-stationary environments, making it more applicable to real-world scenarios like robotics and autonomous systems.
- **Uncertainty-Aware Learning:** The integration of evidential learning into reinforcement learning opens new avenues for uncertainty-aware modeling, which can improve robustness and adaptability in dynamic settings.
- **Practical Applications:** The findings are particularly relevant for applications where environments are inherently non-stationary, such as robotic systems experiencing wear and tear or autonomous agents navigating changing terrains.
- **Future Research:** The success of EPPO suggests that probabilistic and uncertainty-aware methods could be further explored to enhance other reinforcement learning algorithms and address similar challenges in the field.

---

This paper makes a significant contribution to the field of reinforcement learning by introducing a novel approach to maintaining plasticity in non-stationary environments, with potential implications for both theoretical advancements and practical applications.

