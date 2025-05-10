import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import pandas as pd
import numpy as np
import glob
import argparse
import time
from datetime import datetime

from src.value_iteration import ValueIterationAgent
from src.direct_estimation import DirectEstimationAgent
from src.qlearning import QLearningAgent
from src.qlearning import CustomCliffWalkingWrapper
from src.reinforce import ReinforceAgent
from src.utils.evaluator import evaluate_policy
from src.utils.plotter import draw_rewards


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

def run_value_iteration_experiment(exp_dir):
    print("\n--------------------------")
    print("  🤖 Value Iteration 🤖 ")
    print("--------------------------\n")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.95]: "))
    num_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))
    epsilon = float(input("Enter epsilon value for convergence [e.g. 0.001]: "))

    # Create environment and agent 
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = ValueIterationAgent(env, gamma=gamma, epsilon=epsilon)

    # Create latest directory for temporary files
    latest_dir = os.path.join("experiments", "valueIteration", "latest")
    os.makedirs(latest_dir, exist_ok=True)

    print("\n🚀 Training in progress...")
    start_time = time.time()
    agent.train()
    training_time = time.time() - start_time

    policy = agent.policy()
    print("\n🎯 Learned Policy:")
    print(agent.print_policy(policy))

    # Save the learned policy
    policy_path = os.path.join(exp_dir, "learned_policy.txt")
    save_policy(policy, policy_path)

    # Evaluate the policy using the evaluator
    results = evaluate_policy(env, policy, num_episodes=num_episodes, algo_dir="valueIteration")

    # Move evaluation metrics file to experiment directory
    eval_metrics_file = os.path.join(latest_dir, "evaluation_metrics.csv")
    if os.path.exists(eval_metrics_file):
        os.rename(eval_metrics_file, os.path.join(exp_dir, "episode_metrics.csv"))
    
    # Remove latest directory
    if os.path.exists(latest_dir):
        os.rmdir(latest_dir)
    
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


def run_direct_estimation_experiment(exp_dir):
    print("\n---------------------------")
    print("  🤖 Direct Estimation 🤖 ")
    print("----------------------------\n")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.95]: "))
    num_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))
    num_trajectories = int(input("Number of trajectories for sampling [e.g. 500]: "))
    max_iters = int(input("Maximum iterations for training [e.g. 1000]: "))
    patience = int(input("Patience for convergence [e.g. 100]: "))
    num_runs = int(input("Number of runs to execute [e.g. 5]: "))

    # Create environment
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)

    # Store metrics from all runs
    all_metrics = []

    # Run multiple experiments
    for run in range(num_runs):
        run_dir = os.path.join(exp_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)

        # Create agent for this run
        agent = DirectEstimationAgent(env, gamma=gamma, num_trajectories=num_trajectories, max_iters=max_iters, patience=patience)

        # Create latest directory for temporary files
        latest_dir = os.path.join("experiments", "directEstimation", "latest")
        os.makedirs(latest_dir, exist_ok=True)

        print(f"\n🚀 Training run {run + 1}/{num_runs} in progress...")
        start_time = time.time()
        agent.train()
        training_time = time.time() - start_time

        policy = agent.policy()
        print(f"\n🎯 Learned Policy (Run {run + 1}):")
        print(agent.print_policy(policy))

        # Save the learned policy
        policy_path = os.path.join(run_dir, "learned_policy.txt")
        save_policy(policy, policy_path)

        # Evaluate the policy using the evaluator
        results = evaluate_policy(env, policy, num_episodes=num_episodes, algo_dir="directEstimation")

        # Move evaluation metrics file to run directory
        eval_metrics_file = os.path.join(latest_dir, "evaluation_metrics.csv")
        if os.path.exists(eval_metrics_file):
            os.rename(eval_metrics_file, os.path.join(run_dir, "episode_metrics.csv"))
        
        # Remove latest directory
        if os.path.exists(latest_dir):
            os.rmdir(latest_dir)
        
        print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
        print(f"🎯 Success rate: {results['success_rate']:.2%}")
        print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
        print(f"⚡ Training time: {training_time:.2f} seconds")

        # Save metrics for this run
        metrics = {
            "run": run,
            "gamma": gamma,
            "num_trajectories": num_trajectories,
            "max_iters": max_iters,
            "patience": patience,
            "mean_reward": results['mean_return'],
            "mean_steps": results['mean_steps'],
            "success_rate": results['success_rate'],
            "training_time": training_time
        }
        
        # Save individual run metrics
        save_metrics("metrics.csv", metrics, run_dir)

        # Generate and save rewards plot for this run
        plot_path = os.path.join(run_dir, "rewards_plot.png")
        draw_rewards(results['rewards'], plot_path)

        # Add metrics to list for summary
        all_metrics.append(metrics)

    # Create summary metrics file in experiment directory
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

    # Print summary statistics
    print("\n📊 Summary Statistics Across All Runs:")
    print(f"Mean reward: {summary_df['mean_reward'].mean():.2f} ± {summary_df['mean_reward'].std():.2f}")
    print(f"Mean success rate: {summary_df['success_rate'].mean():.2%} ± {summary_df['success_rate'].std():.2%}")
    print(f"Mean steps: {summary_df['mean_steps'].mean():.2f} ± {summary_df['mean_steps'].std():.2f}")
    print(f"Mean training time: {summary_df['training_time'].mean():.2f} ± {summary_df['training_time'].std():.2f} seconds")

def run_qlearning_experiment(exp_dir):
    print("\n--------------------------")
    print("  🤖 Q-Learning 🤖 ")
    print("--------------------------\n")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.95]: "))
    num_episodes = int(input("Number of episodes for training [e.g. 2000]: "))
    learning_rate = float(input("Enter learning rate [e.g. 0.1]: "))
    epsilon = float(input("Enter initial epsilon value [e.g. 0.9]: "))
    decay = float(input("Enter epsilon decay rate [e.g. 0.95]: "))
    t_max = int(input("Enter maximum steps per episode [e.g. 250]: "))
    eval_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))
    penalty = float(input("Enter penalty for moving to the right [e.g. -2]: "))
    num_runs = int(input("Number of runs to execute [e.g. 5]: "))

    # Create environment
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    env = CustomCliffWalkingWrapper(env, penalty=penalty)

    # Store metrics from all runs
    all_metrics = []

    # Run multiple experiments
    for run in range(num_runs):
        run_dir = os.path.join(exp_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)

        # Create agent for this run
        agent = QLearningAgent(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon, t_max=t_max, decay=decay)

        # Create latest directory for temporary files
        latest_dir = os.path.join("experiments", "qlearning", "latest")
        os.makedirs(latest_dir, exist_ok=True)

        print(f"\n🚀 Training run {run + 1}/{num_runs} in progress...")
        start_time = time.time()
        
        # Train the agent
        rewards = []
        for i in range(num_episodes):
            reward = agent.learn_from_episode(i)
            rewards.append(reward)
            
        training_time = time.time() - start_time

        policy = agent.policy()
        print(f"\n🎯 Learned Policy (Run {run + 1}):")
        print(agent.print_policy(policy))

        # Save the learned policy
        policy_path = os.path.join(run_dir, "learned_policy.txt")
        save_policy(policy, policy_path)

        # Evaluate the policy using the evaluator
        results = evaluate_policy(env, policy, num_episodes=eval_episodes, algo_dir="qlearning")

        # Move evaluation metrics file to run directory
        eval_metrics_file = os.path.join(latest_dir, "evaluation_metrics.csv")
        if os.path.exists(eval_metrics_file):
            os.rename(eval_metrics_file, os.path.join(run_dir, "episode_metrics.csv"))
        
        # Remove latest directory
        if os.path.exists(latest_dir):
            os.rmdir(latest_dir)
        
        print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
        print(f"🎯 Success rate: {results['success_rate']:.2%}")
        print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
        print(f"⚡ Training time: {training_time:.2f} seconds")

        # Save metrics for this run
        metrics = {
            "run": run,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "initial_epsilon": epsilon,
            "epsilon_decay": decay,
            "t_max": t_max,
            "training_episodes": num_episodes,
            "left_penalty": penalty,
            "mean_reward": results['mean_return'],
            "mean_steps": results['mean_steps'],
            "success_rate": results['success_rate'],
            "training_time": training_time
        }
        
        # Save individual run metrics
        save_metrics("metrics.csv", metrics, run_dir)

        # Generate and save rewards plot for this run
        plot_path = os.path.join(run_dir, "rewards_plot.png")
        draw_rewards(results['rewards'], plot_path)

        # Add metrics to list for summary
        all_metrics.append(metrics)

    # Create summary metrics file in experiment directory
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

    # Print summary statistics
    print("\n📊 Summary Statistics Across All Runs:")
    print(f"Mean reward: {summary_df['mean_reward'].mean():.2f} ± {summary_df['mean_reward'].std():.2f}")
    print(f"Mean success rate: {summary_df['success_rate'].mean():.2%} ± {summary_df['success_rate'].std():.2%}")
    print(f"Mean steps: {summary_df['mean_steps'].mean():.2f} ± {summary_df['mean_steps'].std():.2f}")
    print(f"Mean training time: {summary_df['training_time'].mean():.2f} ± {summary_df['training_time'].std():.2f} seconds")

def run_reinforce_experiment(exp_dir):
    print("\n--------------------------")
    print("  🤖 REINFORCE 🤖 ")
    print("--------------------------\n")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.9]: "))
    learning_rate = float(input("Enter learning rate [e.g. 0.99]: "))
    lr_decay = float(input("Enter learning rate decay [e.g. 0.99]: "))
    training_episodes = int(input("Number of episodes for training [e.g. 1000]: "))
    t_max = int(input("Enter maximum steps per episode [e.g. 200]: "))
    eval_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))
    num_runs = int(input("Number of runs to execute [e.g. 5]: "))

    # Create environment
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)

    # Store metrics from all runs
    all_metrics = []

    # Run multiple experiments
    for run in range(num_runs):
        run_dir = os.path.join(exp_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)

        # Create agent for this run
        agent = ReinforceAgent(env, gamma=gamma, learning_rate=learning_rate, lr_decay=lr_decay, training_episodes=training_episodes)

        # Create latest directory for temporary files
        latest_dir = os.path.join("experiments", "reinforce", "latest")
        os.makedirs(latest_dir, exist_ok=True)

        print(f"\n🚀 Training run {run + 1}/{num_runs} in progress...")
        start_time = time.time()
        
        # Train the agent
        rewards = []
        losses = []
        for i in range(training_episodes):
            reward, loss = agent.learn_from_episode()
            rewards.append(reward)
            losses.append(loss)
            
        training_time = time.time() - start_time

        policy, _ = agent.policy()
        print(f"\n🎯 Learned Policy (Run {run + 1}):")
        print(agent.print_policy(policy))

        # Save the learned policy
        policy_path = os.path.join(run_dir, "learned_policy.txt")
        save_policy(policy, policy_path)

        # Evaluate the policy using the evaluator
        results = evaluate_policy(env, policy, num_episodes=eval_episodes, algo_dir="reinforce")

        # Move evaluation metrics file to run directory
        eval_metrics_file = os.path.join(latest_dir, "evaluation_metrics.csv")
        if os.path.exists(eval_metrics_file):
            os.rename(eval_metrics_file, os.path.join(run_dir, "episode_metrics.csv"))
        
        # Remove latest directory
        if os.path.exists(latest_dir):
            os.rmdir(latest_dir)
        
        print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
        print(f"🎯 Success rate: {results['success_rate']:.2%}")
        print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
        print(f"⚡ Training time: {training_time:.2f} seconds")

        # Save metrics for this run
        metrics = {
            "run": run,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "learning_rate_decay": lr_decay,
            "training_episodes": training_episodes,
            "mean_reward": results['mean_return'],
            "mean_steps": results['mean_steps'],
            "success_rate": results['success_rate'],
            "training_time": training_time,
            "final_learning_rate": agent.learning_rate,
            "mean_loss": np.mean(losses)
        }
        
        # Save individual run metrics
        save_metrics("metrics.csv", metrics, run_dir)

        # Generate and save rewards plot for this run
        plot_path = os.path.join(run_dir, "rewards_plot.png")
        draw_rewards(results['rewards'], plot_path)

        # Add metrics to list for summary
        all_metrics.append(metrics)

    # Create summary metrics file in experiment directory
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

    # Print summary statistics
    print("\n📊 Summary Statistics Across All Runs:")
    print(f"Mean reward: {summary_df['mean_reward'].mean():.2f} ± {summary_df['mean_reward'].std():.2f}")
    print(f"Mean success rate: {summary_df['success_rate'].mean():.2%} ± {summary_df['success_rate'].std():.2%}")
    print(f"Mean steps: {summary_df['mean_steps'].mean():.2f} ± {summary_df['mean_steps'].std():.2f}")
    print(f"Mean training time: {summary_df['training_time'].mean():.2f} ± {summary_df['training_time'].std():.2f} seconds")
    print(f"Mean final learning rate: {summary_df['final_learning_rate'].mean():.6f} ± {summary_df['final_learning_rate'].std():.6f}")
    print(f"Mean loss: {summary_df['mean_loss'].mean():.6f} ± {summary_df['mean_loss'].std():.6f}")

def clear_files():
    experiments_dir = "experiments"
    if os.path.exists(experiments_dir):
        for algo_dir in ["valueIteration", "directEstimation", "qlearning", "reinforce"]:
            algo_path = os.path.join(experiments_dir, algo_dir)
            if os.path.exists(algo_path):
                # Only remove directories that start with "experiment_"
                for exp_dir in os.listdir(algo_path):
                    if exp_dir.startswith("experiment_"):
                        exp_path = os.path.join(algo_path, exp_dir)
                        if os.path.isdir(exp_path):
                            # First remove files in run directories for direct estimation
                            if algo_dir == "directEstimation":
                                for run_dir in os.listdir(exp_path):
                                    run_path = os.path.join(exp_path, run_dir)
                                    if os.path.isdir(run_path):
                                        for file in os.listdir(run_path):
                                            os.remove(os.path.join(run_path, file))
                                        os.rmdir(run_path)
                            # Remove remaining files and the experiment directory itself
                            for file in os.listdir(exp_path):
                                file_path = os.path.join(exp_path, file)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                                elif os.path.isdir(file_path):
                                    # Remove any remaining subdirectories (should only happen for direct estimation)
                                    for subfile in os.listdir(file_path):
                                        os.remove(os.path.join(file_path, subfile))
                                    os.rmdir(file_path)
                            os.rmdir(exp_path)
                # Only remove the algorithm directory if it's empty
                if not os.listdir(algo_path):
                    os.rmdir(algo_path)
        print("🧹 Default experiment files have been cleared!")
    else:
        print("📂 No experiment files found.")

def select_algorithm():
    print("\n🤖 Available Algorithms:")
    print("1. Value Iteration")
    print("2. Direct Estimation")
    print("3. Q-Learning")
    print("4. REINFORCE")
    while True:
        try:
            choice = int(input("\nSelect algorithm: "))
            if choice in [1, 2, 3, 4]:
                return choice
            print("❌ Please enter 1, 2, 3, or 4")
        except ValueError:
            print("❌ Please enter a valid number")

def main():
    parser = argparse.ArgumentParser(description="🤖 Cliff Walking Experiment Runner")
    parser.add_argument('--clean', action='store_true', help='Clear all experiment files')
    args = parser.parse_args()

    if args.clean:
        clear_files()
    else:
        print("\n🎮 === Cliff Walking Experiment Runner === 🤖")
        # Select algorithm
        algorithm = select_algorithm()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if algorithm == 1:
            algo_dir = "valueIteration"
            run_experiment = run_value_iteration_experiment
        elif algorithm == 2:
            algo_dir = "directEstimation"
            run_experiment = run_direct_estimation_experiment
        elif algorithm == 3:
            algo_dir = "qlearning"
            run_experiment = run_qlearning_experiment
        else:  # algorithm == 4
            algo_dir = "reinforce"
            run_experiment = run_reinforce_experiment
        
        exp_dir = os.path.join("experiments", algo_dir, f"experiment_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

        # Set up logging to file
        log_filename = os.path.join(exp_dir, "experiment_log.txt")
        sys.stdout = Logger(log_filename)
        
        # Run the selected experiment
        run_experiment(exp_dir)
        
        # Restore original stdout
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
