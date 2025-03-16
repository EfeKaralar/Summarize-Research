# An Optimal Control View of Adversarial Machine Learning

**Authors:** Xiaojin Zhu

**Published:** 2018-11-11

**URL:** http://arxiv.org/abs/1811.04422v1

## Abstract

I describe an optimal control view of adversarial machine learning, where the
dynamical system is the machine learner, the input are adversarial actions, and
the control costs are defined by the adversary's goals to do harm and be hard
to detect. This view encompasses many types of adversarial machine learning,
including test-item attacks, training-data poisoning, and adversarial reward
shaping. The view encourages adversarial machine learning researcher to utilize
advances in control theory and reinforcement learning.

## Summary

### Summary of "An Optimal Control View of Adversarial Machine Learning" by Xiaojin Zhu (2018)

---

### 1. Key Findings
- **Optimal Control Framework**: The paper proposes an optimal control framework to model adversarial machine learning, where the machine learner is treated as a dynamical system, adversarial actions are control inputs, and control costs reflect the adversary's goals (e.g., causing harm while remaining undetected).
- **Unified View**: This framework unifies various adversarial machine learning scenarios, including test-time attacks, training-data poisoning, and adversarial reward shaping, under a single theoretical lens.
- **Interdisciplinary Insights**: The approach encourages leveraging advances in control theory and reinforcement learning to address adversarial machine learning challenges.
- **Formalization of Adversarial Goals**: The paper formalizes adversarial objectives (e.g., minimal perturbation for test-time attacks or targeted model manipulation for poisoning) as control problems with specific cost functions.
- **Defense Implications**: The framework also provides insights into defense strategies, such as adversarial training, by interpreting them through the lens of control theory.

---

### 2. Research Question/Problem
The paper addresses the problem of understanding and formalizing adversarial machine learning from a unified theoretical perspective. Specifically, it seeks to answer:
- How can adversarial machine learning be modeled systematically?
- What mathematical foundations (e.g., optimal control) can be applied to analyze and mitigate adversarial attacks?
- How can insights from control theory and reinforcement learning inform the design of robust machine learning systems?

---

### 3. Methodology
The author adopts an **optimal control framework** to model adversarial machine learning:
- **Dynamical System**: The machine learner is treated as a dynamical system, with its state evolving based on adversarial inputs (control actions).
- **Control Inputs**: Adversarial actions (e.g., modifying training data or test inputs) are modeled as control inputs.
- **Cost Functions**: The adversary's objectives (e.g., causing harm, minimizing detection) are formalized as control costs, including running costs (step-by-step effort) and terminal costs (final outcome).
- **Problem Formulation**: The adversarial problem is framed as an optimization problem, minimizing the total control cost subject to system dynamics and constraints.
- **Examples**: The framework is applied to specific adversarial scenarios, such as training-data poisoning, test-time attacks, and adversarial reward shaping, to illustrate its generality.

---

### 4. Results
- **Training-Data Poisoning**: The framework formalizes poisoning attacks as a control problem, where the adversary manipulates training data to induce a desired model. The problem is shown to be equivalent to a Stackelberg game or bi-level optimization.
- **Test-Time Attacks**: Test-time attacks are modeled as a one-step control problem, where the adversary perturbs a test input to mislead the model while minimizing perturbation effort.
- **Adversarial Reward Shaping**: The framework extends to scenarios where the adversary manipulates reward signals in reinforcement learning, influencing the learner's behavior.
- **Defense Strategies**: The paper highlights how defense mechanisms, such as adversarial training, can be interpreted within the control framework, emphasizing the importance of robustness to adversarial inputs.

---

### 5. Implications
- **Theoretical Foundation**: The optimal control view provides a rigorous mathematical foundation for adversarial machine learning, enabling systematic analysis and comparison of different attack and defense strategies.
- **Interdisciplinary Synergy**: By bridging machine learning and control theory, the paper encourages cross-disciplinary research, potentially leading to novel solutions for adversarial challenges.
- **Practical Applications**: The framework can guide the design of robust machine learning systems by formalizing adversarial objectives and constraints, helping practitioners anticipate and mitigate attacks.
- **Future Research**: The paper opens avenues for exploring advanced control-theoretic techniques (e.g., robust control, reinforcement learning) in adversarial machine learning, potentially leading to more effective defenses.

---

In summary, Xiaojin Zhu's paper provides a novel and unifying perspective on adversarial machine learning by framing it as an optimal control problem. This approach not only deepens the theoretical understanding of adversarial attacks but also offers practical insights for developing robust machine learning systems.

