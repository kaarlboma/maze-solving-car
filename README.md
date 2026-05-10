# Maze Solving Car

A reinforcement learning project implementing Q-Learning and Deep Q-Network (DQN) to solve grid-based mazes — a stepping stone toward deploying a trained agent on a physical RC car.

## What's here

- **GridWorld** — a simple grid environment with walls, a start position, and a goal
- **Q-Learning** — tabular Q-learning baseline
- **DQN** — neural network-based Q-learning using PyTorch and an experience replay buffer
- **Visualizer** — pygame GUI to train and watch the agent navigate the maze

## How to run

Install dependencies:
```bash
pip install pygame numpy torch
```

Run the visualizer:
```bash
python visualize.py
```

Toggle between Q-Learning and DQN using the **Mode** button in the sidebar, then click **Train**.

To run DQN training headlessly:
```bash
python test_dqn.py
```
