# Reversal Generative Reinforcement Learning
## Introduction
We introduce **R**eversal **G**enerative **R**einforcement **L**earning (**RGRL**), a simple **model-free** and **value-function-free** reinforcement learning method, where **only** neural networks are involved.

**RGRL** treats an agent as ***a neural net ensemble*** where ***state*** and ***action*** are ***input*** datum, and ***reward*** is ***output*** data. First, when the neural net ensemble (the agent) observes certain state, it initializes and updates its actions through error backpropagation, with given desired reward as target. Second, the agent executes the updated actions and learns the ensuing consequence (state, executed actions and actual reward). Third, based on the learned neural net ensemble, the agent starts observing new state and initializing or updating its actions again. Through this iterative cycle of trial and error, the agent gradually develops a belief that approximates the true environment, enabling it to discover the optimal solution for maximizing rewards. This approach is very similar to reinforcement learning in our intuitive finding. Since action is gradually generated from input layer rather than output layer, we refer to this method as **R**eversal **G**enerative **R**einforcement **L**earning (**RGRL**).

In **RGRL**, we use deep neural nets to substitute Bellman function to provide more flexibility. That is to say, we use deep neural nets to map future reward to present state and actions or to predict future reward using present state and actions.

Our previous research can be seen in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), where we simply used supervised learning method rather than reinforcement learning method. However, in **RGRL**, we have inherited the spirit of the previous research while incorporating the concept of reinforcement learning, allowing the agent to learn from the ensuing consequence of it deduced or updated actions. In practice, this enables the agent to find the optimal solution to achieve the maximum reward in the environment more quickly.

## Details
For more detials, please refer to RGRL's technical [paper](https://github.com/Brownwang0426/RGRL/blob/main/paper.pdf).

## Features
- **Only Neural Nets are involved**: Really.
- **Highly custimizable**: All you need to do is to customize state, action and reward-shaping or vectorizing. You can also make state as raw input if you prefer :-) Why not? It is deep neural network :-) 

## Future Works
- **Online learning**: For the present time, we present offline learning version of RGRL.

## Getting Started
To get started with RGRL, follow these steps:

1. **Open the train.ipynb in colab or local environment**
2. **Set up configs in the Control Board section according to the instructions thereof**
3. **Select T4 GPU or above, Ctrl + F10, restart and Ctrl + F10**
4. **Take a rest and wait for the result**

## Experimental Results
We use traditional **Cartpole** as an example and show that how the size of an ensemble of neural networks affect the overall performace of the agent.

## Why an ensemble of neural networks rather than a single neural network?

Building on our previous research in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), we observed that when using error backpropagation to update input data for model inversion in a trained deep neural network—similar to techniques in DeepDream—the updated input data often becomes unstable and gets stuck. After multiple trials, we identified that this instability occurs because the input is essentially performing gradient descent on a single error surface generated by one deep neural network, leading to numerous local minima, regardless of the network’s level of training.

To mitigate this instability, we borrow the concept of traditional stochastic gradient descent method, where input data and labels are shuffled to train a neural network. However, the difference now is that we shuffled an ensemble of trained deep neural networks during the input data update, preventing the input from getting trapped in local minima.

In our earlier work, we demonstrated the effectiveness of this method, achieving a 97.3% success rate in solving blank Sudoku puzzles using the ensemble of deep neural networks.

## Project Structure

We try to keep the structure as clean and easy as possible. The whole structure of RGRL is as follows:

```bash
RGRL/
│
├── envs/                    # environement-related files
│   ├── __init__.py          # package indicator
│   ├── env_cartpole.py      # environement-related file for cartpole, such as vectorizing state, actions and reward
│   ├── ...                  # environement-related file for ...
│   └── env_lunarlander.py   # environement-related file for lunarlander, such as vectorizing state, actions and reward
│
├── models/                  # model implementation files
│   ├── __init__.py          # package indicator
│   ├── model_att.py         # model implementation file for attention
│   └── model_rnn.py         # model implementation file for rnn, gru, lstm
│
├── utils/                   # utility functions
│   ├── __init__.py          # package indicator
│   ├── util_att.py          # utility function for attention
│   └── util_rnn.py          # utility function for rnn, gru, lstm
│
├── train.py                 # main script for training the model
│
├── paper.pdf                # main technical paper for understanding the mechansim behind RGRL
│
├── LICENSE                  # license and user agreement, etc.
│
└── README.md                # project documentation
```

## Status
The project is currently in active development. We are continually adding new features and improving the performance.

## License
RGRL is released under the [MIT](https://github.com/Brownwang0426/RGRL/blob/main/LICENSE) license.

## Related Works
- [Deducing Decision by Error Backpropagation](https://ala2022.github.io/papers/ALA2022_paper_4.pdf)
- [Modeling and Optimization of Complex Building Energy Systems with Deep Neural Networks](https://ieeexplore.ieee.org/document/8335578)
- [Offline Contextual Bandits for Wireless Network Optimization](https://arxiv.org/abs/2111.08587)





