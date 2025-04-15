import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        
    def calc_action_value(self, state, action):
        action_value = sum([prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.unwrapped.P[state][action]]) 
        return action_value

    def select_action(self, state):
        best_action = best_value = None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if not best_value or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):  
                state_values.append(self.calc_action_value(state, action))
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    
    def policy(self):   
        policy = np.zeros(self.env.observation_space.n) 
        for s in range(self.env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy