# Genrl
## Introduction
We introduce **Gen**erative **R**einforcement **L**earning (**Genrl**), a **model-free** and **value-function-free** reinforcement learning method, where **only** neural networks are involved.

**Genrl** treats an agent as ***a neural net ensemble*** where ***state*** and ***action*** are ***input*** datum, and ***reward*** is ***output*** data. First, when the neural net ensemble (the agent) observes certain state, it initializes and updates its actions through error backpropagation, with given desired reward as target. Second, the agent executes the updated actions and learns the ensuing consequence (state, executed actions and actual reward). Under this iterative updating-then-learning process, the agent gradually forms a belief approximating the real environement, allowing the agent to find the optimal solution to achieve the maximum reward in the environment. This approach is very similar to reinforcement learning in our intuitive finding. Since action is gradually generated from input layer rather than output layer, we refer to this method as **Gen**erative **R**einforcement **L**earning (**Genrl**).

In **Genrl**, you don't need Bellman function or other value funtions to map reward to action because the deep neural net **IS** the Bellman function!

Our previous research can be seen in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), where we simply used supervised learning method rather than reinforcement learning method. However, in **Genrl**, we have inherited the spirit of the previous research while incorporating the concept of reinforcement learning, allowing the agent to learn from the ensuing consequence of it deduced or updated actions. This enables the agent to find the optimal solution to achieve the maximum reward in the environment more quickly.

## Features
- **Neural Nets are all you need**: No need for the Bellman function or other value funtions to map reward to action, and none of the complicated jargon found in current deep reinforcement learning methods.
- **Highly custimizable**: All you need to do is to customize state, action and reward-shaping or vectorizing. You can also make state as raw input if you prefer :-) Why not? It is deep neural network :-)

## Future Works
- **Online learning**: For the present time, we present offline learning version of Genrl. However, we are still working on the online learning version :-)

## Getting Started
To get started with Genrl, follow these steps:

1. **Open the .ipynb in colab**
2. **Ctrl + F10**
3. **Take a rest and wait for the result**

## Experimental Results
We use traditional **Cartpole** as an example and show that how the size of an ensemble of neural networks affect the overall performace of the agent.



## Why an ensemble of neural networks rather than a single neural network?
Based on our previous research in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), we found that when using error backpropagation to update input data for model inversion in a trained deep neural network—similar to techniques used in DeepDream—the updated input data often becomes unstable and gets stuck. After several attempts, we identified that this instability arises because the input data is factually performing gradient descent on a single error surface created by one deep neural network, which results in numerous local minima, regardless of how well-trained the network is.

To address this instability, we adapted the traditional stochastic gradient descent approach of shuffling input data and labels to train a deep neural network. However, this time,  we shuffle an ensemble of trained deep neural networks during the updating of the input data to ensure that the updated input data won't get stuck at local minima.

Our previous [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf) showed it is quite successful since this method has an ensemble of trained deep neural networks solve completely blank Sudokus with 97.3% success rate.

## Status
The project is currently in active development. We are continually adding new features and improving the performance.

## License
Genrl is released under the [MIT](https://github.com/Brownwang0426/Genrl/blob/main/LICENSE) license.

## Related Works
- [Rewarded Region Replay (R3) for Policy Learning with Discrete Action Space](https://arxiv.org/pdf/2405.16383)
- [Optimal Policy Sparsication and Low Rank Decomposition for Deep Reinforcement Learning](https://arxiv.org/pdf/2403.06313)





