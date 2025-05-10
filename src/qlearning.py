import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper

# Constants
SLIPPERY = True
T_MAX = 200
NUM_EPISODES = 2000
GAMMA = 0.95
LEARNING_RATE = 0.1
EPSILON = 0.9
RENDER_MODE = "ansi"
MIN_EPSILON = 0.1
MIN_LEARNING_RATE = 0.1
EPSILON_DECAY = 0.95
LEARNING_RATE_DECAY = 0.95

class QLearningAgent:
    """
    A Q-Learning agent implementation for the CliffWalking environment.
    Learns the optimal policy through experience using Q-learning algorithm.
    """
    def __init__(self, env, gamma, learning_rate, epsilon, t_max, ep_decay, lr_decay):
        """
        Initialize the Q-Learning agent.

        Args:
            env: Gymnasium environment instance
            gamma (float): Discount factor for future rewards
            learning_rate (float): Learning rate for Q-value updates
            epsilon (float): Exploration probability for epsilon-greedy policy
            t_max (int): Maximum number of timesteps per episode
        """
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max
        self.decay = ep_decay
        self.lr_decay = lr_decay

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (int): Current state index
            training (bool): Whether the agent is in training mode

        Returns:
            int: Selected action
        """
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state,])
        
    def update_Q(self, state, action, reward, next_state):
        """
        Update Q-values using the Q-learning update rule.

        Args:
            state (int): Current state index
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state index
        """
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error
        
    def learn_from_episode(self, num_episode):
        """
        Run one episode of learning, updating Q-values based on experience.

        Returns:
            float: Total reward accumulated in the episode
        """
        self.epsilon = max(MIN_EPSILON, self.epsilon * (self.decay ** num_episode))
        self.learning_rate = max(MIN_LEARNING_RATE, self.learning_rate * (self.lr_decay ** num_episode))
        state, _ = self.env.reset()
        total_reward = 0
        for i in range(self.t_max):
            action = self.select_action(state)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state)
            if is_done:
                break
            state = new_state
        return total_reward

    def policy(self):
        """
        Extract the current policy from learned Q-values.

        Returns:
            numpy.ndarray: Array of optimal actions for each state
        """
        policy = np.zeros(self.env.observation_space.n) 
        for s in range(self.env.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))        
        return policy
    
    def print_policy(self, policy):
        """
        Print a visual representation of the policy using arrows.

        Args:
            policy (numpy.ndarray): Array of actions representing the policy
        """
        visual_help = {0:'^', 1:'>', 2:'v', 3:'<'}
        policy_arrows = [visual_help[x] for x in policy]
        print(np.array(policy_arrows).reshape([4, 12]))
    

class CustomCliffWalkingWrapper(Wrapper):
    """
    A custom wrapper for the Cliff Walking environment that modifies the reward structure.
    """
    def __init__(self, env, penalty):
        """
        Initialize the wrapper.

        Args:
            env: Gymnasium environment instance
        """
        super().__init__(env)
        self.penalty = penalty
    
    def step(self, action):
        """
        Modified step function that implements custom rewards.

        Args:
            action (int): Action to take

        Returns:
            tuple: (state, reward, is_done, truncated, info)
        """
        state, reward, is_done, truncated, info = self.env.step(action)
        if action == 3:
            reward = self.penalty
        return state, reward, is_done, truncated, info