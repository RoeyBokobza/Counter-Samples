# Counter-Samples: A Stateless Strategy <br> to Neutralize Black Box Adversarial Attacks
## Overview
Counter-Samples is a novel defense against black box attacks, where attackers use the victim model as an oracle to craft their adversarial examples. Unlike traditional preprocessing defenses that rely on sanitizing input samples, our stateless strategy counters the attack process itself. For every query we evaluate a counter-sample instead, where the counter-sample is the original sample optimized against the attacker's objective. By countering every black box query with a targeted white box optimization, our strategy effectively introduces an asymmetry to the game to the defender's advantage. This defense not only effectively misleads the attacker's search for an adversarial example, it also preserves the model's performance on legitimate inputs unlike past works. 

## Introduction
Adversarial examples pose a significant threat to deep neural networks by subtly altering inputs to mislead model predictions. In this work, we focus on defending against query-based black-box attacks, where attackers iteratively query a model to optimize adversarial perturbations. Our proposed method introduces a novel counter-sample preprocessor that misguides attackers during the query process. For each input, we generate a counter-sample that maintains the original class's features, effectively confining the attacker's search within the class manifold. This approach can drastically increases the number of queries required for a successful attack, while preserving model performance on legitimate data. The figure below illustrates how the defense misleads the attacker during each query iteration.
![Intro_figure3 (1)_page-0001](https://github.com/user-attachments/assets/4d6765f2-271b-45f0-b7e3-7b45edba3c6a)

## Methodology

The preprocessor transforms an input sample `x` into a counter-sample `x*` while preserving the model's original prediction. This transformation alters the local loss surface, misleading the attacker's search direction and complicating the adversarial process.

The counter-sample `x*` is iteratively optimized using gradient descent, defined by the following equation: <br>
x*<sub>t+1</sub> = x*<sub>t</sub> - α ∇<sub>x*<sub>t</sub></sub> L(f(x*<sub>t</sub>;θ), ŷ) <br>

where `f` is the target model, `α` is the step size, ∇<sub>x*<sub>t</sub></sub> L is the gradient of the loss function, and `ŷ` is the predicted label. This process continues until the counter-sample `x*` converges.

To enhance robustness against more advanced attacks, we add a small amount of Gaussian noise `z ∼ N(μ, σ²)` to each query before applying the transformation. The final defense mechanism is given by:
`f(x*) = f(T(x + z); θ)`

**Key Features:**

 - **Preservation of clean task performance**: Counter-Samples maintains the model's accuracy on legitimate inputs, ensuring no degradation in performance.
 - **State-of-the-art (SOTA) defense**: Counter-Samples demonstrates superior performance against SOTA black-box query-based attacks compared to other preprocessors. Additionally, it outperforms Adversarial Training (AT) in overall results.



