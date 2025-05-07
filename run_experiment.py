import gymnasium as gym
import pandas as pd
import numpy as np
import os
import glob
import argparse
import sys
import time
from datetime import datetime
from src.value_iteration import ValueIterationAgent  # agent file
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

def save_metrics(filename, data):
    os.makedirs("experiments", exist_ok=True)
    path = os.path.join("experiments", filename)
    pd.DataFrame([data]).to_csv(path, index=False)
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
    start_time = time.time()
    rewards, max_diffs = agent.train()
    training_time = time.time() - start_time

    policy = agent.policy()
    print("\n🎯 Learned Policy:")
    print(agent.print_policy(policy))

    # Evaluate the policy using the new evaluator
    results = evaluate_policy(env, policy, num_episodes=num_episodes)
    
    print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
    print(f"🎯 Success rate: {results['success_rate']:.2%}")
    print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
    print(f"⚡ Training time: {training_time:.2f} seconds")

    # Save metrics 💾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"value_iteration_metrics_{timestamp}.csv"
    
    metrics = {
        "mean_reward": results['mean_return'],
        "mean_steps": results['mean_steps'],
        "success_rate": results['success_rate'],
        "training_time": training_time,
        "gamma": gamma
    }
    
    metrics_path = save_metrics(metrics_filename, metrics)
    print(f"\n📊 Metrics saved in: {metrics_path}")

def clear_files():
    """Clear all experiment files 🧹"""
    # Clear CSV files
    csv_files = glob.glob('experiments/*.csv')
    txt_files = glob.glob('experiments/*.txt')
    
    files = csv_files + txt_files
    if files:
        for file in files:
            os.remove(file)
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
        # Set up logging to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join("experiments", f"value_iteration_log_{timestamp}.txt")
        os.makedirs("experiments", exist_ok=True)
        sys.stdout = Logger(log_filename)
        
        print("\n")
        print("🎮 === Cliff Walking Experiment Runner === 🤖")
        print("🚀 Let's explore the cliff walking environment! 🌟\n")
        run_value_iteration_experiment()
        
        # Restore original stdout
        sys.stdout = sys.__stdout__
