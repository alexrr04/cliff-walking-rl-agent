import gymnasium as gym
import pandas as pd
import numpy as np
import os
import glob
import argparse
import sys
import time
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from src.value_iteration import ValueIterationAgent 
from src.utils.evaluator import evaluate_policy

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_policy(policy, save_path):
    """Save the policy in a readable format"""
    visual_help = {0: '^', 1: '>', 2: 'v', 3: '<'}
    policy_arrows = [visual_help[int(x)] for x in policy]
    policy_grid = np.array(policy_arrows).reshape([4, 12])
    
    with open(save_path, 'w') as f:
        f.write("Learned Policy (^ = up, > = right, v = down, < = left):\n\n")
        for row in policy_grid:
            f.write(' '.join(row) + '\n')

def save_metrics(filename, data, exp_dir):
    """Save metrics to the experiment directory"""
    path = os.path.join(exp_dir, filename)
    pd.DataFrame([data]).to_csv(path, index=False)
    return path

def draw_rewards(rewards, save_path):
    """Plot and save the rewards obtained during testing."""
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

def run_value_iteration_experiment():
    print("🌟====================🌟")
    print("  🤖 Value Iteration 🤖 ")
    print("🌟====================🌟")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.95]: "))
    num_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))
    epsilon = float(input("Epsilon value for exploration [e.g. 1e-8]: "))

    # Create experiment directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    value_iter_dir = os.path.join("experiments", "valueIteration")
    exp_dir = os.path.join(value_iter_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Set up logging to file
    log_filename = os.path.join(exp_dir, "experiment_log.txt")
    sys.stdout = Logger(log_filename)

    # Create environment and agent 
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = ValueIterationAgent(env, gamma=gamma, epsilon=epsilon)

    print("\n🚀 Training in progress...")
    start_time = time.time()
    rewards, max_diffs = agent.train()
    training_time = time.time() - start_time

    policy = agent.policy()
    print("\n🎯 Learned Policy:")
    print(agent.print_policy(policy))

    # Save the learned policy
    policy_path = os.path.join(exp_dir, "learned_policy.txt")
    save_policy(policy, policy_path)

    # Evaluate the policy using the new evaluator
    results = evaluate_policy(env, policy, num_episodes=num_episodes)
    
    print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
    print(f"🎯 Success rate: {results['success_rate']:.2%}")
    print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
    print(f"⚡ Training time: {training_time:.2f} seconds")

    # Save metrics 
    metrics = {
        "gamma": gamma,
        "epsilon": epsilon,
        "mean_reward": results['mean_return'],
        "mean_steps": results['mean_steps'],
        "success_rate": results['success_rate'],
        "training_time": training_time
    }
    
    save_metrics("metrics.csv", metrics, exp_dir)

    # Generate and save rewards plot 
    plot_path = os.path.join(exp_dir, "rewards_plot.png")
    draw_rewards(results['rewards'], plot_path)

    # Restore original stdout
    sys.stdout = sys.__stdout__

def clear_files():
    """Clear all experiment files 🧹"""
    value_iter_dir = os.path.join("experiments", "valueIteration")
    if os.path.exists(value_iter_dir):
        for exp_dir in os.listdir(value_iter_dir):
            exp_path = os.path.join(value_iter_dir, exp_dir)
            if os.path.isdir(exp_path):
                for file in os.listdir(exp_path):
                    os.remove(os.path.join(exp_path, file))
                os.rmdir(exp_path)
        os.rmdir(value_iter_dir)
        print("🧹 All experiment files have been cleared!")
    else:
        print("📂 No experiment files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🤖 Cliff Walking Experiment Runner")
    parser.add_argument('--clear', action='store_true', help='Clear all experiment files')
    args = parser.parse_args()

    if args.clear:
        clear_files()
    else:
        print("🎮 === Cliff Walking Experiment Runner === 🤖")
        run_value_iteration_experiment()
