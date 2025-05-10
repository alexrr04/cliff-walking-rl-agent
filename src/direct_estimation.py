import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections

# Constants
SLIPPERY = True
T_MAX = 200
NUM_EPISODES = 100
NUM_TRAJECTORIES = 500
GAMMA = 0.95
RENDER_MODE = "ansi"
MAX_ITERS = 500
PATIENCE = 100    # iteraciones sin mejora para parar

class DirectEstimationAgent:
    def __init__(self, env, gamma, num_trajectories, max_iters, patience):
        """
        Initialize the Value Iteration agent.

        Args:
            env: Gymnasium environment instance
            gamma (float): Discount factor for future rewards
        """
        self.env = env
        self.state, _ = self.env.reset()

        # Para R(s,a,s')
        self.rewards_sum = collections.defaultdict(float)
        self.rewards_count = collections.defaultdict(int)

        # Para T(s,a,s')
        self.transits = collections.defaultdict(collections.Counter)

        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_trajectories = num_trajectories
        self.max_iters = max_iters
        self.patience = patience

    def play_n_random_steps(self, count):
        """
        Play random steps in the environment to gather experience.

        Args:
            count (int): Number of random steps to take

        Updates the rewards and transitions dictionaries with observed data.
        """
        self.state, _ = self.env.reset()

        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards_sum[(self.state, action, new_state)] += reward
            self.rewards_count[(self.state, action, new_state)] += 1
            self.transits[(self.state, action)][new_state] += 1
            if is_done or truncated:
                self.state, _ = self.env.reset() 
            else: 
                self.state = new_state

    def calc_action_value(self, state, action):
        """
        Calculate the value of taking an action in a given state.

        Args:
            state (int): Current state index
            action (int): Action to evaluate

        Returns:
            float: Expected value of taking the action in the state
        """
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        if total == 0:
            return 0.0
        action_value = 0.0
        for s_, count in target_counts.items():
            r = self.rewards_sum[(state, action, s_)] / self.rewards_count[(state, action, s_)]
            prob = (count / total)
            action_value += prob*(r + self.gamma * self.V[s_])
        return action_value

    def select_action(self, state):
        """
        Select the best action for a given state based on current value estimates.

        Args:
            state (int): Current state index

        Returns:
            int: Best action to take
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        """
        Perform one iteration of the value iteration algorithm.
        Updates state values based on the Bellman equation.

        Returns:
            tuple: (Updated value function array, Maximum value difference)
        """
        self.play_n_random_steps(self.num_trajectories)
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    
    def policy(self):
        """
        Extract the optimal policy from the learned value function.

        Returns:
            numpy.ndarray: Array of optimal actions for each state
        """   
        policy = np.zeros(self.env.observation_space.n) 
        for s in range(self.env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy

    
    def check_improvements(self):
        """
        Test the current agent policy over multiple episodes.

        Returns:
            float: Average reward across test episodes
        """
        reward_test = 0.0
        for i in range(NUM_EPISODES):
            total_reward = 0.0
            state, _ = self.env.reset()
            for i in range(T_MAX):
                action = self.select_action(state)
                new_state, new_reward, is_done, truncated, _ = self.env.step(action)
                total_reward += new_reward
                if is_done: 
                    break
                state = new_state
            reward_test += total_reward
        reward_avg = reward_test / NUM_EPISODES
        return reward_avg

    def train(self): 
        rewards = []
        max_diffs = []
        t = 0
        best_reward = -np.inf
        max_diff = 1.0
        no_improve = 0

        while t < self.max_iters:
            _, max_diff = self.value_iteration()
            max_diffs.append(max_diff)
            print("After value iteration, max_diff = " + str(max_diff))
            t += 1
            reward_test = self.check_improvements()
            rewards.append(reward_test)
                
            if reward_test > best_reward:
                print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
                best_reward = reward_test
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"Sin mejora en {self.patience} iteraciones. Parando.")
                break

        return rewards, max_diffs

    def print_policy(self, policy):
        """
        Print a visual representation of the policy using arrows.

        Args:
            policy (numpy.ndarray): Array of actions representing the policy
        """
        visual_help = {0:'^', 1:'>', 2:'v', 3:'<'}
        policy_arrows = [visual_help[x] for x in policy]
        print(np.array(policy_arrows).reshape([4, 12]))


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

# # Initialize the environment
# env = gym.make("CliffWalking-v0", render_mode=RENDER_MODE, is_slippery=SLIPPERY)

# # Initialize and train the agent
# agent = DirectEstimationAgent(env, gamma=GAMMA, num_trajectories=NUM_TRAJECTORIES)
# rewards, max_diffs = train(agent)

# # Compute and print agent's policy
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