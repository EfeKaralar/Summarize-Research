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
- **Optimal Control Framework**: The paper proposes viewing adversarial machine learning through the lens of optimal control theory, where the machine learning system is treated as a dynamical system, and adversarial actions are control inputs.
- **Unified Perspective**: This framework unifies various adversarial machine learning scenarios, including training-data poisoning, test-time attacks, and adversarial reward shaping, under a single theoretical umbrella.
- **Control Costs**: Adversarial actions are modeled with control costs that reflect the adversary's dual goals of causing harm and remaining undetected.
- **Bridging Fields**: The paper encourages leveraging advances in control theory and reinforcement learning to address challenges in adversarial machine learning.
- **Formalization of Attacks**: The framework provides a formal structure to model and analyze adversarial attacks, enabling a deeper understanding of their dynamics and potential defenses.

---

### 2. Research Question/Problem
The paper addresses the problem of understanding and formalizing adversarial machine learning attacks within a unified theoretical framework. Traditional machine learning assumes independent and identically distributed (i.i.d.) data, but adversarial attacks introduce non-i.i.d., structured perturbations. The research question is: *How can adversarial machine learning be systematically modeled and analyzed using principles from optimal control theory?*

---

### 3. Methodology
The author adopts an **optimal control framework** to model adversarial machine learning:
- **System Dynamics**: The machine learning system is treated as a dynamical system, with states representing the model (e.g., classifier weights) and control inputs representing adversarial actions (e.g., poisoned data or perturbed inputs).
- **Control Costs**: Adversarial actions are associated with running costs (e.g., effort to modify data) and terminal costs (e.g., harm caused by the attack).
- **Problem Formulation**: The adversarial problem is framed as an optimization problem, where the adversary seeks to minimize control costs while achieving their objectives.
- **Examples**: The framework is applied to specific adversarial scenarios, including:
  - **Training-data poisoning**: Adversary modifies training data to corrupt the learned model.
  - **Test-time attacks**: Adversary perturbs test inputs to cause misclassification.
  - **Adversarial reward shaping**: Adversary manipulates rewards in reinforcement learning to influence the learned policy.

---

### 4. Results
- **Unified Framework**: The optimal control view successfully formalizes diverse adversarial machine learning scenarios, providing a common language and structure for analysis.
- **Insights into Adversarial Dynamics**: The framework reveals that adversarial attacks can be understood as control problems with specific cost structures and constraints.
- **Potential for Defense Strategies**: The control-theoretic perspective suggests that defense mechanisms (e.g., adversarial training) can also be framed as control problems, offering new avenues for developing robust machine learning systems.
- **Algorithmic Connections**: The paper highlights connections to existing techniques in control theory (e.g., dynamic programming, Pontryagin’s minimum principle) and machine learning (e.g., bi-level optimization, Stackelberg games).

---

### 5. Implications
- **Theoretical Foundation**: The paper provides a rigorous mathematical foundation for adversarial machine learning, moving beyond ad hoc analyses of specific attacks.
- **Interdisciplinary Insights**: By bridging control theory and machine learning, the framework encourages cross-disciplinary research, potentially leading to novel algorithms and defenses.
- **Practical Applications**: The formalization of adversarial attacks as control problems can guide the development of more robust machine learning systems, particularly in security-critical domains.
- **Future Research**: The paper opens new research directions, such as exploring stochastic and continuous-time control formulations for adversarial machine learning and leveraging reinforcement learning techniques for adversarial defense.

---

In summary, Xiaojin Zhu's paper offers a novel and unifying perspective on adversarial machine learning by framing it as an optimal control problem. This approach not only deepens the theoretical understanding of adversarial attacks but also provides practical tools for analyzing and mitigating them.

