import gymnasium as gym
import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime
from src.value_iteration import ValueIterationAgent  # agent file
from src.utils.evaluator import evaluate_policy

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

    # Evaluate the policy using the new evaluator
    results = evaluate_policy(env, policy, num_episodes=num_episodes)
    
    # Save results 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training progress
    train_filename = f"value_iteration_training_{timestamp}.csv"
    train_path = save_results(train_filename, {
        "iteration": list(range(1, len(max_diffs)+1)),
        "max_diff": max_diffs
    })
    print(f"📁 Training results saved in: {train_path}")
    
    # Save evaluation results
    eval_filename = f"value_iteration_evaluation_{timestamp}.csv"
    eval_path = save_results(eval_filename, {
        "episode": list(range(1, len(results['rewards'])+1)),
        "reward": results['rewards']
    })
    print(f"📁 Evaluation results saved in: {eval_path}")

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
