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
NUM_EPISODES = 500
GAMMA = 0.95
LEARNING_RATE = 0.5
EPSILON = 0.9
RENDER_MODE = "ansi"
MIN_EPSILON = 0.2
DECAY = 0.9

class QLearningAgent:
    """
    A Q-Learning agent implementation for the CliffWalking environment.
    Learns the optimal policy through experience using Q-learning algorithm.
    """
    def __init__(self, env, gamma, learning_rate, epsilon, t_max):
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
        
    def learn_from_episode(self):
        """
        Run one episode of learning, updating Q-values based on experience.

        Returns:
            float: Total reward accumulated in the episode
        """
        
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
    

class CustomCliffWalkingWrapper(Wrapper):
    """
    A custom wrapper for the Cliff Walking environment that modifies the reward structure.
    """
    def __init__(self, env):
        """
        Initialize the wrapper.

        Args:
            env: Gymnasium environment instance
        """
        super().__init__(env)
    
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
            reward = -2
        return state, reward, is_done, truncated, info

def draw_rewards(rewards):
    """
    Plot the rewards obtained during training/testing.

    Args:
        rewards (list): List of rewards to plot
    """
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    
def print_policy(policy):
    """
    Print a visual representation of the policy using arrows.

    Args:
        policy (numpy.ndarray): Array of actions representing the policy
    """
    visual_help = {0:'^', 1:'>', 2:'v', 3:'<'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))

def rollout(env, policy, max_steps=300):
    """
    Execute one episode with the greedy policy.

    Returns
    -------
    reached_goal : bool
    steps        : int
    total_return : float
    """
    state, _ = env.reset()
    total_return = 0.0
    for t in range(1, max_steps + 1):
        # print(env.render())               # returns an ASCII string
        # r, c = divmod(state, 12)
        # print(f"t={t:3d}  state=({r},{c})  index={state:2d}")

        action = int(policy[state])
        state, reward, is_done, truncated, _ = env.step(action)
        total_return += reward

        if is_done:                        # reached [3,11]
            print(f"\n🎉  Goal reached in {t} steps, return = {total_return}\n")
            return True, t, total_return
        if truncated:                         # hit the TimeLimit wrapper
            break

    print("\n💥  Episode ended without reaching the goal\n")
    return False, t, total_return

env = gym.make("CliffWalking-v0", render_mode=RENDER_MODE, is_slippery=SLIPPERY)

env = CustomCliffWalkingWrapper(env)
agent = QLearningAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, t_max=T_MAX)
rewards = []
for i in range(100):
    agent.epsilon = max(MIN_EPSILON, EPSILON * (DECAY ** i))
    reward = agent.learn_from_episode()
    print("New reward: " + str(reward))
    rewards.append(reward)
# draw_rewards(rewards)

policy = agent.policy()
print_policy(policy)

is_done = False
rewards = []
for n_ep in range(NUM_EPISODES):
    state, _ = env.reset()
    print('Episode: ', n_ep)
    total_reward = 0
    for i in range(T_MAX):
        action = agent.select_action(state, training=False)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        env.render()
        if is_done:
            break
    rewards.append(total_reward)
draw_rewards(rewards)

successes = 0
steps_mean = 0
rewards_count = 0
episodes = NUM_EPISODES

for ep in range(episodes):
    print(f"\n=== Episode {ep} ===")
    reached_goal, steps, G = rollout(env, policy)
    successes += int(reached_goal)
    steps_mean += steps
    rewards_count += G

steps_mean /= episodes
print(f"\nSuccess rate: {successes}/{episodes}, Mean steps: {steps_mean:.2f}, Mean return: {rewards_count/episodes:.2f}")