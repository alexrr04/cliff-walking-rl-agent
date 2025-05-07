import gymnasium as gym
import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime
from src.value_iteration import ValueIterationAgent  # agent file
from src.utils.evaluator import rollout


def evaluate_policy(env, policy, num_episodes=100, max_steps=200):
    total_reward = 0.0
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = int(policy[state])
            state, reward, done, *_ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_reward += episode_reward
    return total_reward / num_episodes

def save_results(filename, data):
    os.makedirs("experiments", exist_ok=True)
    path = os.path.join("experiments", filename)
    pd.DataFrame(data).to_csv(path, index=False)
    return path

def run_value_iteration_experiment():

    print("🌟====================🌟")
    print("  🤖 Value Iteration 🤖 ")
    print("🌟====================🌟")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.95]: "))
    num_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))

    # Create environment and agent 🎮
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = ValueIterationAgent(env, gamma=gamma)

    print("\n🚀 Training in progress...")
    rewards, max_diffs = agent.train()

    policy = agent.policy()
    print("\n🎯 Learned Policy:")
    print(agent.print_policy(policy))
    avg_reward = evaluate_policy(env, policy, num_episodes=num_episodes)

    print(f"\n🏆 Average reward after training: {avg_reward:.2f}")

    # Save results 💾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"value_iteration_results_{timestamp}.csv"
    save_path = save_results(filename, {
        "iteration": list(range(1, len(max_diffs)+1)),
        "max_diff": max_diffs
    })
    print(f"📁 Results saved in: {save_path}")

def clear_csv_files():
    """Clear all CSV files in the experiments directory 🧹"""
    csv_files = glob.glob('experiments/*.csv')
    if csv_files:
        for file in csv_files:
            os.remove(file)
        print("🧹 All CSV files have been cleared from experiments directory!")
    else:
        print("📂 No CSV files found in experiments directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🤖 Cliff Walking Experiment Runner")
    parser.add_argument('--clear', action='store_true', help='Clear all CSV files in experiments directory')
    args = parser.parse_args()

    if args.clear:
        clear_csv_files()
    else:
        print("\n")
        print("🎮 === Cliff Walking Experiment Runner === 🤖")
        run_value_iteration_experiment()
