# Genrl
## Introduction
We introduce **Gen**erative **R**einforcement **L**earning (**Genrl**), a **model-free** and **value-function-free** reinforcement learning method, where **only** neural networks are involved.

**Genrl** treats an agent as ***a neural net ensemble*** where ***state*** and ***action*** are ***input*** datum, and ***reward*** is ***output*** data. Firstly, when the neural net ensemble (the agent) confronts certain state, it updates its actions through error backpropagation, with given desired reward as target. Secondly, the agent executes the updated actions and learns the ensuing consequence (state, executed actions and obtained reward). Under this iterative updating-then-learning process, the agent gradually forms a belief approximating the real environement, allowing the agent to find the optimal solution to achieve the maximum reward in the environment. This approach is very similar to reinforcement learning in our intuitive finding. Since action is gradually generated from input layer rather than output layer, we refer to this method as **Gen**erative **R**einforcement **L**earning (**Genrl**).

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
Suppose, for each step, an agent is an ensemble of neural networks `𝕎` where `𝕎 = {W₁, W₂, ..., Wₘ}`, present state is represented by `s`, intitial actions are represented by `a` where `a = {a₁, a₂, ...,aₜ}`, and desired reward is represented by `r'`. Then we have the following iterative update for `a`:

![image](https://github.com/user-attachments/assets/e2c721cc-c34a-4d76-95cc-e824d01a5e22)


```pseudo
1. **Initialize** long-term memory `𝔇`
2. **Initialize** neural ensemble 𝕎 where `𝕎 = { W₁ , W₂ , ... , Wₘ }`
3. **Initialize** desired reward `r'`
4. **For** each episode:
    1. **Initialize** environment
    2. **Initialize** short-term memory `D`
    3. **While** not done:
        1. **Observe** state `s`
        2. **Initialize** actions `a` where `a = { a₁ , a₂ , ... , aₜ }`
        3. **For** each iteration:
            1. **Select** `Wᵢ` from `𝕎`
            2. **Perform** back-propagation:  
               $a ← a - β * ( ∂/∂a ) E( r' , f( Wᵢ , (s, a) ) )$
        4. **Execute** action `a₁` where `a₁ ∈ a`
        5. **Observe** reward `r`
        6. **Store** `s , a₁ , r` to `D`
    4. **End While**
    5. **Return** agent performance to human for inspection
    6. **Sequentialize** `D` to `𝔇`
    7. **For** each `Wᵢ` in `𝕎`:
        1. **For** each iteration:
            1. **Select** `sⱼ , aⱼ , rⱼ` from `𝔇`  
               where `aⱼ = { aⱼ₁ , aⱼ₂ , ... , aⱼₜ }`
            2. **Perform** back-propagation:  
               `Wᵢ ← Wᵢ - α * ( ∂/∂Wᵢ ) E( rⱼ , f( Wᵢ , ( sⱼ , aⱼ ) ) )`
        2. **End For**
    8. **End For**
5. **End For**
Initialize short term memory $ D $
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
Based on our previous research in this [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf), we discovered that when using error backpropagation to update input data to perform some kind of **model inversion mechanism** in a trained deep neural network similar to those in **deep dream**, the updated input data is always unstable and prone to get stuck. After several attempts, we finally discovered the reason behind this instability is that the input data is factually doing gradient descent upon a single error surface created by a single deep neural network, which leads to tons of local minima, no matter how well-trained the deep neural network is.

To negate this instability, we mimic the traditional stochastic gradient descent where we shuffle x and y to train a deep neural network, but this time we shuffle an ensemble of trained deep neural networks during the updating of the input data to ensure that the updated input data won't get stuck at local minima.

Our previous [paper](https://ala2022.github.io/papers/ALA2022_paper_4.pdf) showed it is quite successful since our method have an ensemble of trained deep neural networks solve a completely blank Sudoku with 97.3% success rate.

## Experimental Results
We use traditional **Cartpole** as an example and show that how the size of an ensemble of neural networks affect the overall performace of the agent.



Also, since our method solely involves deep neural networks, the **catastrophic forgetting** issue in deep learning has to be addressed. To overcome this defect, we also applied **E**lastic **W**eight **C**ontrol (**EWC**) in this [paper](https://arxiv.org/pdf/1612.00796). It helps.

## Status
The project is currently in active development. We are continually adding new features and improving the performance.

## License
Genrl is released under the [MIT](https://github.com/Brownwang0426/Genrl/blob/main/LICENSE) license.
