import numpy as np
from GridWorld import GridWorld
from agent_dqn import DQN, ReplayBuffer, train_dqn

ACTIONS = ['U', 'D', 'L', 'R']
REWARDS = {'-': -1, 'G': 10}

world_arr = np.array([
    ['-', '-', '-'],
    ['-', 'X', '-'],
    ['-', '-', 'G']
])

world = GridWorld(world_arr, (0, 0), REWARDS, 0.9)
agent = DQN(ACTIONS)
buffer = ReplayBuffer(capacity=1000)

train_dqn(world, agent, buffer, episodes=1000)
print("Training complete\n")

print("Learned policy:")
for r in range(world.rows):
    row = ""
    for c in range(world.cols):
        cell = world_arr[r][c]
        if cell == 'X':
            row += "  X "
        elif cell == 'G':
            row += "  G "
        else:
            action = agent.choose_action(epsilon=0, state=(r, c))
            row += f"  {action} "
    print(row)
