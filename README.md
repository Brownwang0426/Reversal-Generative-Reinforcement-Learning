# Genrl
## Introduction
We introduce **Gen**erative **R**einforcement **L**earning (**Genrl**), a **model-free** and **value-function-free** reinforcement learning method, where **only** neural networks are involved.

**Genrl** treats an agent as ***a neural net ensemble*** where ***state*** and ***action*** are input datum, and ***reward*** is output data. Firstly, when the neural net ensemble (the agent) confronts certain state, it updates its actions through error backpropagation, with given desired reward as target. Secondly, the agent executes the updated actions and learns the ensuing consequence (state, executed actions and obtained reward). Under this iterative updating-then-learning process, the agent gradually forms a belief approximating the real environement, allowing the agent to find the optimal solution to achieve the maximum reward in the environment. This approach is very similar to reinforcement learning in our intuitive finding. Since action is gradually generated from input layer rather than output layer, we refer to this method as **Gen**erative **R**einforcement **L**earning (**Genrl**).

Our previous research can be seen in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), where we simply used supervised learning method rather than reinforcement learning method. However, in **Genrl**, we have inherited the spirit of the previous research while incorporating the concept of reinforcement learning, allowing the agent to learn from the ensuing consequence of it deduced or updated actions. This enables the agent to find the optimal solution to achieve the maximum reward in the environment more quickly.

## Features
- **Neural Nets are all you need**: No need for the Bellman function or other value funtions, and none of the complicated jargon found in current deep reinforcement learning methods.
- **Highly custimizable**: All you need to do is to customize state, action and reward-shaping or vectorizing.
- **Both online and offline learning**: We provide both online and offline learning versions. However, currently the online version is more time costly so we will focus on the offline learning version.

## Getting Started
To get started with Genrl, follow these steps:

1. **Open the .ipynb in colab**
2. **Ctrl + F10**
3. **Take a rest and wait for the result**

## Algorithm
Suppose an agent is an ensemble of neural networks $\mathbbm{W}$ where $\mathbbm{W} = \{ W_1 , W_2 , \ldots , W_m \}$. $\acute{r}$ is desired reward, $\acute{r}$ is present state


```pseudo
function binary_search(arr, target)
    low <- 0
    high <- length(arr) - 1

    while low <= high do
        mid <- floor((low + high) / 2)
        
        if arr[mid] = target then
            return mid
        else if arr[mid] < target then
            low <- mid + 1
        else
            high <- mid - 1

    return -1
```

## Why an ensemble of neural networks rather than a single neural network?
Based on our previous research in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), we discovered that when using error backpropagation to update input data like:

$    \hat{x}\leftarrow\hat{x}-\beta\frac{\partial}{\partial\hat{x}}{E}\Big(y,f(\hat{x})\Big) $

## Experimental Results
We use Cartpole as example.

## Status
The project is currently in active development. We are continually adding new features and improving the performance.


## License
Genrl is released under the [MIT](https://github.com/Brownwang0426/Genrl/blob/main/LICENSE) license.
