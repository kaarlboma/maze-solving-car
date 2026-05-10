import numpy as np
import random

class GridWorld:
    def __init__(self, world, start, rewards_dict, discount):
        self.start = start
        self.world = world
        self.current_pos = start
        self.rewards_dict = rewards_dict
        self.rows = world.shape[0]
        self.cols = world.shape[1]
        self.discount = discount

    def step(self, action):
        row, col = self.current_pos
        done = False
        if action == 'U' and (row - 1 >= 0) and (self.world[row - 1][col] != 'X'):
            self.current_pos = (row - 1, col)
            if self.world[self.current_pos[0]][self.current_pos[1]] == 'G':
                done = True
            square = self.world[self.current_pos[0]][self.current_pos[1]]
            reward = self.rewards_dict[square]
            return (self.current_pos, reward, done)
        elif action == 'D' and (row + 1 < self.rows) and (self.world[row + 1][col] != 'X'):
            self.current_pos = (row + 1, col)
            if self.world[self.current_pos[0]][self.current_pos[1]] == 'G':
                done = True
            square = self.world[self.current_pos[0]][self.current_pos[1]]
            reward = self.rewards_dict[square]
            return (self.current_pos, reward, done)
        elif action == 'L' and (col - 1 >= 0) and (self.world[row, col - 1] != 'X'):
            self.current_pos = (row, col - 1)
            if self.world[self.current_pos[0]][self.current_pos[1]] == 'G':
                done = True
            square = self.world[self.current_pos[0]][self.current_pos[1]]
            reward = self.rewards_dict[square]
            return (self.current_pos, reward, done)
        elif action == 'R' and (col + 1 < self.cols) and (self.world[row][col + 1] != 'X'):
            self.current_pos = (row, col + 1)
            if self.world[self.current_pos[0]][self.current_pos[1]] == 'G':
                done = True
            square = self.world[self.current_pos[0]][self.current_pos[1]]
            reward = self.rewards_dict[square]
            return (self.current_pos, reward, done)
        return (self.current_pos, -1, done)
    
    def reset(self):
        self.current_pos = self.start

class Agent:
    def __init__(self, world, actions):
        self.qtable = {}
        self.actions = actions
        for i in range(world.rows):
            for j in range(world.cols):
                for action in actions:
                    self.qtable[((i, j), action)] = 0
    
    def choose_action(self, epsilon, state):
        if random.random() < epsilon:
            selected_action = random.choice(self.actions)
        else:
            best_qval = self.qtable[(state, 'U')]
            selected_action = 'U'
            for action in self.actions:
                if self.qtable[(state, action)] > best_qval:
                    best_qval = self.qtable[(state, action)]
                    selected_action = action
        return selected_action
            
    def update(self, state, action, reward, next_state, alpha, gamma):
        """
        state is current state
        action is action selected
        reward is immediate reward of being in current state
        next_state is the next state after choosing the action from current state
        alpha is learning rate
        gamma is discount factor
        """
        best_qval = self.qtable[(next_state, 'U')]
        for a in self.actions:
            if self.qtable[(next_state, a)] > best_qval:
                best_qval = self.qtable[(next_state, a)]
        self.qtable[(state, action)] = self.qtable[(state, action)] + (alpha * (reward + (gamma * best_qval) - self.qtable[(state, action)]))
        
def train(world, agent, max_iter = 1000, epsilon = 0.3, episodes = 1000, alpha = 0.1, gamma = 0.9):
    for i in range(episodes):
        done = False
        world.reset()
        current_state = world.current_pos
        while not done:
            action = agent.choose_action(epsilon, current_state)
            next_state, reward, done = world.step(action)
            agent.update(current_state, action, reward, next_state, alpha, gamma)
            current_state = next_state
    return agent