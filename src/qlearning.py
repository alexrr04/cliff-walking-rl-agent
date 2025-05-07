import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper

# Constants
SLIPPERY = True
T_MAX = 250
NUM_EPISODES = 2000
GAMMA = 0.95
LEARNING_RATE = 0.1
EPSILON = 0.9
RENDER_MODE = "ansi"
MIN_EPSILON = 0.1
DECAY = 0.95

class QLearningAgent:
    """
    A Q-Learning agent implementation for the CliffWalking environment.
    Learns the optimal policy through experience using Q-learning algorithm.
    """
    def __init__(self, env, gamma, learning_rate, epsilon, t_max, decay):
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
        self.decay = decay

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
        self.epsilon = max(MIN_EPSILON, self.epsilon * (self.decay** num_episode))
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



# def draw_rewards(rewards):
#     """
#     Plot the rewards obtained during training/testing.

#     Args:
#         rewards (list): List of rewards to plot
#     """
#     data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(x='Episode', y='Reward', data=data)

#     plt.title('Rewards Over Episodes')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.grid(True)
#     plt.tight_layout()

#     plt.show()

# def rollout(env, policy, max_steps=300):
#     """
#     Execute one episode with the greedy policy.

#     Returns
#     -------
#     reached_goal : bool
#     steps        : int
#     total_return : float
#     """
#     state, _ = env.reset()
#     total_return = 0.0
#     for t in range(1, max_steps + 1):
#         # print(env.render())               # returns an ASCII string
#         # r, c = divmod(state, 12)
#         # print(f"t={t:3d}  state=({r},{c})  index={state:2d}")

#         action = int(policy[state])
#         state, reward, is_done, truncated, _ = env.step(action)
#         total_return += reward

#         if is_done:                        # reached [3,11]
#             print(f"\n🎉  Goal reached in {t} steps, return = {total_return}\n")
#             return True, t, total_return
#         if truncated:                         # hit the TimeLimit wrapper
#             break

#     print("\n💥  Episode ended without reaching the goal\n")
#     return False, t, total_return


# env = gym.make("CliffWalking-v0", render_mode=RENDER_MODE, is_slippery=SLIPPERY)

# env = CustomCliffWalkingWrapper(env)
# agent = QLearningAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, t_max=T_MAX)
# rewards = []

# # Train the agent
# for i in range(NUM_EPISODES):
#     reward = agent.learn_from_episode(i)
#     print("New reward: " + str(reward))
#     rewards.append(reward)
# # draw_rewards(rewards)

# policy = agent.policy()
# print_policy(policy)


# # Test the agent once it is trained
# test_rewards = []
# successes = 0
# total_steps = 0
# total_reward = 0
# episodes = NUM_EPISODES

# for ep in range(episodes):
#     print(f"\n=== Episode {ep} ===")
#     render_episode = ep == 0  
#     reached_goal, steps, G = rollout(env, policy, max_steps=T_MAX)
    
#     test_rewards.append(G)
#     successes += int(reached_goal)
#     total_steps += steps
#     total_reward += G

# success_rate = successes / episodes
# mean_steps = total_steps / episodes
# mean_return = total_reward / episodes

# print(f"\n✅ Evaluación completa:")
# print(f"Success rate: {successes}/{episodes} = {success_rate:.2%}")
# print(f"Mean steps per episode: {mean_steps:.2f}")
# print(f"Mean return per episode: {mean_return:.2f}")

# draw_rewards(test_rewards)
