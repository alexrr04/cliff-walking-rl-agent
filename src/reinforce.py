import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gymnasium import Wrapper

SLIPPERY = False
TRAINING_EPISODES = 1000
NUM_EPISODES = 500
GAMMA = 0.9
T_MAX = 200
LEARNING_RATE = 0.99    
LEARNING_RATE_DECAY = 0.99
RENDER_MODE = "ansi"

class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, lr_decay=1, seed=0):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        # Objeto que representa la política (J(theta)) como una matriz estados X acciones,
        # con una probabilidad inicial para cada par estado accion igual a: pi(a|s) = 1/|A|
        self.policy_table = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        np.random.seed(seed)

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training:
            # Escogemos la acción según el vector de policy_table correspondiente a la acción,
            # con una distribución de probabilidad igual a los valores actuales de este vector
            return np.random.choice(np.arange(self.env.action_space.n), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        loss = -np.sum(np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)

        # policy_logits = np.log(self.policy_table)
        for t in range(len(states)):

            s, a = states[t], actions[t]

            G_t = discounted_rewards[t]

            probs = self.policy_table[s]

            # Move every action logit a bit down …
            self.policy_table[s] += self.learning_rate * G_t * (-probs)
            # … and the chosen action a bit up (+1 in the one-hot)
            self.policy_table[s, a] += self.learning_rate * G_t

            # numerical safety & renormalisation
            self.policy_table[s] = np.clip(self.policy_table[s], 1e-8, None)
            self.policy_table[s] /= self.policy_table[s].sum()

            # # reconstrucción de π(a|s;θ) desde los logits
            # action_probs = np.exp(policy_logits[states[t]])
            # action_probs /= np.sum(action_probs)

            # # ∇ log π(aₜ|sₜ) para softmax es (1 - π(aₜ|sₜ))
            # policy_gradient = G_t * (1 - action_probs[actions[t]])

            # # ascenso de gradiente:
            # policy_logits[states[t], actions[t]] += self.learning_rate * policy_gradient

            # Alternativa:
            # policy_gradient = 1.0 / action_probs[actions[t]]
            # policy_logits[states[t], actions[t]] += self.learning_rate * G_t * policy_gradient

        # # re-normalización a probabilidades
        # exp_logits = np.exp(policy_logits)
        # self.policy_table = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    def learn_from_episode(self):
        state, _ = self.env.reset()
        episode = []
        done = False
        step = 0
        total_reward = 0
        while not done and step < T_MAX:
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            step += 1
        loss = self.update_policy(zip(*episode))
        self.learning_rate = self.learning_rate * self.lr_decay
        return total_reward, loss

    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table


def draw_history(history, title):
    window_size = 50
    data = pd.DataFrame({'Episode': range(1, len(history) + 1), title: history})
    data['rolling_avg'] = data[title].rolling(window_size).mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y=title, data=data)
    sns.lineplot(x='Episode', y='rolling_avg', data=data)

    plt.title(title + ' Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(title)
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
    print(np.array(policy_arrows).reshape([4, 12]))

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
agent = ReinforceAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, lr_decay=LEARNING_RATE_DECAY)

rewards = []
losses = []
for i in range(TRAINING_EPISODES):
    reward, loss = agent.learn_from_episode()
    policy, policy_table = agent.policy()
    print(policy_table)
    print(f"Last reward: {reward}, last loss: {loss}, new lr: {agent.learning_rate}")
    print_policy(policy)
    print(f"End of iteration [{i + 1}/{TRAINING_EPISODES}]")
    rewards.append(reward)
    losses.append(loss)

draw_history(rewards, "Reward")
draw_history(losses, "Loss")

# Test the agent once it is trained
test_rewards = []
successes = 0
total_steps = 0
total_reward = 0
episodes = NUM_EPISODES

for ep in range(episodes):
    print(f"\n=== Episode {ep} ===")
    render_episode = ep == 0
    reached_goal, steps, G = rollout(env, policy, max_steps=T_MAX)

    test_rewards.append(G)
    successes += int(reached_goal)
    total_steps += steps
    total_reward += G

success_rate = successes / episodes
mean_steps = total_steps / episodes
mean_return = total_reward / episodes

print(f"\n✅ Evaluación completa:")
print(f"Success rate: {successes}/{episodes} = {success_rate:.2%}")
print(f"Mean steps per episode: {mean_steps:.2f}")
print(f"Mean return per episode: {mean_return:.2f}")

draw_rewards(test_rewards)