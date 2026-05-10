from GridWorld import GridWorld
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = []
    
    def push(self, experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)
        
    def sample(self, batch_size):
        return random.sample(self.experiences, batch_size)

class DQN(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def choose_action(self, epsilon, state):
        state = torch.tensor(state, dtype = torch.float32)
        if random.random() < epsilon:
            selected_action = random.choice(self.actions)
        else:
            qvals = self.forward(state)
            action_idx = torch.argmax(qvals).item()
            index_to_action = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
            selected_action = index_to_action[action_idx]
        return selected_action
    
def train_dqn(world, agent, buffer, batch_size = 32, epsilon = 0.3, episodes = 1000, alpha = 0.1, gamma = 0.9):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(agent.parameters(), lr = alpha)
    for i in range(episodes):
        done = False
        world.reset()
        current_state = world.current_pos
        while not done:
            # take one step
            action = agent.choose_action(epsilon, current_state)
            next_state, reward, done = world.step(action)

            # add experience to replay buffer
            experience = (current_state, action, reward, next_state, done)
            buffer.push(experience)

            # sample from replay buffer
            if len(buffer.experiences) >= batch_size:
                sample = buffer.sample(batch_size)

                # converting sample to tensor
                states, actions, rewards, next_states, dones = zip(*sample)
                states = torch.tensor(states, dtype = torch.float32)
                rewards = torch.tensor(rewards, dtype = torch.float32)
                next_states = torch.tensor(next_states, dtype = torch.float32)
                dones = torch.tensor(dones, dtype = torch.bool)

                # converting actions
                action_map = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
                actions = torch.tensor([action_map[a] for a in actions], dtype=torch.long)

                # get predicted q-values
                predicted = agent(states).gather(1, actions.unsqueeze(1))
                target = rewards + gamma * agent(next_states).max(1).values * (~dones).float()

                # train network
                optimizer.zero_grad()
                loss = criterion(predicted.squeeze(1), target)
                loss.backward()
                optimizer.step()
            
            # update current state
            current_state = next_state
    return agent

