# Deep Q-Network (DQN) Agent for CartPole-v1
A complete implementation of a Deep Q-Network (DQN) agent in PyTorch to solve the CartPole-v1 environment.


This project presents a complete implementation of a Deep Q-Network (DQN) agent, built from scratch using PyTorch, to solve the classic CartPole-v1 control problem from Gymnasium. The agent successfully learns a policy to balance a pole on a moving cart.

### Final Result
Here is the fully trained agent demonstrating its learned policy, successfully balancing the pole for the maximum episode duration.
you can watch (cartpole_demo.gif).

## Core Concepts & Skills Demonstrated
Deep Reinforcement Learning: Applied a neural network to approximate the action-value (Q) function.
Experience Replay: Implemented a replay buffer to store and sample past experiences, which breaks the correlation between consecutive samples and stabilizes training.
Fixed Q-Targets: Utilized a separate Target Network to provide stable Q-value targets, preventing the "moving target" problem and significantly improving stability.
Epsilon-Greedy Strategy: Employed an epsilon-greedy policy with a decay schedule for a robust balance between exploration of the environment and exploitation of known good actions.
Hyperparameter Tuning: Methodically selected parameters to ensure efficient and stable learning.
Results & Analysis
The agent was trained for 600 episodes. The environment is considered "solved" when the agent achieves an average reward of 195 or greater over 100 consecutive episodes.
you can see the image(learning_curve.png)

### Key Observations:
Successful Convergence: The agent successfully solved the environment, achieving a final 100-episode average reward of 253.15. It first crossed the "solved" threshold (average reward > 195) around episode 350.
Learning Stability: The learning curve shows the effectiveness of the DQN stability mechanisms. After an initial phase of volatile exploration (episodes 0-300), the 100-episode moving average (red line) shows a consistent and stable upward trend, indicating that the agent is reliably improving its policy.
Performance Saturation: In the later stages of training (episodes 400-600), the agent's performance begins to saturate, consistently achieving high scores, including the maximum possible score of 500.

## Methodology
The agent's architecture consists of two key components:
### Policy Network: A Multi-Layer Perceptron (MLP) that is trained at every step. It takes the environment's state as input and outputs the predicted Q-value for each possible action.
### Target Network: A second MLP with the same architecture. Its weights are a slow-moving average of the policy network's weights (updated via a soft update with TAU = 0.005). This network provides a stable, non-moving target during the loss calculation, which is essential for consistent learning.
The training process uses an epsilon-greedy strategy for action selection, with epsilon decaying from 1.0 to 0.01. All experiences are stored in a replay buffer of size 10,000, and a random minibatch of 64 experiences is sampled at each learning step to update the policy network's weights.


## How to Use This Repository
### 1: Clone the project
### 2: Install dependencies: Install dependencies: pip install -r requirements.txt
### 3: Train a new model: python train.py
### 4: Watch the trained agent play: python evaluate.py

## 🎥 CartPole Demo

![CartPole Demo](https://raw.githubusercontent.com/Umang-projects/DQN-Cartpole/main/cartpole_demo.gif)

