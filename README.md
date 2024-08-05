# Genrl
## Introduction
We introduce **Gen**erative **R**einforcement **L**earning (**Genrl**), a **model-free** and **value-function-free** reinforcement learning method, where **only** neural networks are involved.

**Genrl** treats an agent as ***a neural net ensemble*** where ***state*** and ***action*** are input datum, and ***reward*** is output data. Firstly, when the neural net ensemble (the agent) confronts certain state, it updates its actions through error backpropagation, targeting desired reward. Secondly, the agent executes the updated actions and learns the ensuing consequence (state, executed actions and obtained reward). Under this iterative updating-then-learning process, the agent gradually forms a belief approximating the real environement, allowing the agent to find the optimal solution to achieve the maximum reward in the environment. This approach is very similar to reinforcement learning in biological finding.

Our previous research can be seen in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), where we simply used supervised learning method rather than reinforcement learning method.

However, in **Genrl**, we have inherited the spirit of the previous research while incorporating the concept of reinforcement learning, allowing the agent to learn from the ensuing consequence of it deduced or updated actions. This enables the agent to find the optimal solution to achieve the maximum reward in the environment more quickly.

## Features
- **Neural Nets are all you need**: No need for the Bellman function or other value funtions, and none of the complicated jargon found in current deep reinforcement learning methods.
- **Highly custimizable**: All you need to do is to customize state, action and reward-shaping or vectorizing.

## Status
The project is currently in active development. We are continually adding new features and improving the performance.

## Getting Started
To get started with Genrl, follow these steps:

1. **Open the .ipynb in colab**
2. **Ctrl + F10**
3. **Take a rest and wait for the result**

## License
Genrl is released under the [MIT](https://github.com/Brownwang0426/Genrl/blob/main/LICENSE) license.
