import gymnasium as gym
import pandas as pd
import numpy as np
import os
import glob
import argparse
import sys
import time
from datetime import datetime
from src.value_iteration import ValueIterationAgent
from src.direct_estimation import DirectEstimationAgent
from src.qlearning import QLearningAgent
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


def run_direct_estimation_experiment(exp_dir):
    print("\n---------------------------")
    print("  🤖 Direct Estimation 🤖 ")
    print("----------------------------\n")

    gamma = float(input("Enter gamma value (discount factor) [e.g. 0.95]: "))
    num_episodes = int(input("Number of episodes for evaluation [e.g. 100]: "))
    num_trajectories = int(input("Number of trajectories for sampling [e.g. 500]: "))
    max_iters = int(input("Maximum iterations for training [e.g. 1000]: "))

    # Create environment and agent 
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = DirectEstimationAgent(env, gamma=gamma, num_trajectories=num_trajectories, max_iters=max_iters)

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
    results = evaluate_policy(env, policy, num_episodes=num_episodes)
    
    print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
    print(f"🎯 Success rate: {results['success_rate']:.2%}")
    print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
    print(f"⚡ Training time: {training_time:.2f} seconds")

    # Save metrics 
    metrics = {
        "gamma": gamma,
        "num_trajectories": num_trajectories,
        "max_iters": max_iters,
        "mean_reward": results['mean_return'],
        "mean_steps": results['mean_steps'],
        "success_rate": results['success_rate'],
        "training_time": training_time
    }
    
    save_metrics("metrics.csv", metrics, exp_dir)

    # Generate and save rewards plot 
    plot_path = os.path.join(exp_dir, "rewards_plot.png")
    draw_rewards(results['rewards'], plot_path)

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

    # Create environment and agent 
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = QLearningAgent(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon, t_max=t_max, decay=decay)

    print("\n🚀 Training in progress...")
    start_time = time.time()
    
    # Train the agent
    rewards = []
    for i in range(num_episodes):
        reward = agent.learn_from_episode(i)
        rewards.append(reward)
        
    training_time = time.time() - start_time

    policy = agent.policy()
    print("\n🎯 Learned Policy:")
    print(agent.print_policy(policy))

    # Save the learned policy
    policy_path = os.path.join(exp_dir, "learned_policy.txt")
    save_policy(policy, policy_path)

    # Evaluate the policy using the evaluator
    results = evaluate_policy(env, policy, num_episodes=eval_episodes)
    
    print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
    print(f"🎯 Success rate: {results['success_rate']:.2%}")
    print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
    print(f"⚡ Training time: {training_time:.2f} seconds")

    # Save metrics 
    metrics = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "initial_epsilon": epsilon,
        "epsilon_decay": decay,
        "t_max": t_max,
        "training_episodes": num_episodes,
        "mean_reward": results['mean_return'],
        "mean_steps": results['mean_steps'],
        "success_rate": results['success_rate'],
        "training_time": training_time
    }
    
    save_metrics("metrics.csv", metrics, exp_dir)

    # Generate and save rewards plot 
    plot_path = os.path.join(exp_dir, "rewards_plot.png")
    draw_rewards(results['rewards'], plot_path)

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

    # Create environment and agent 
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = ReinforceAgent(env, gamma=gamma, learning_rate=learning_rate, lr_decay=lr_decay, training_episodes=training_episodes)

    print("\n🚀 Training in progress...")
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
    print("\n🎯 Learned Policy:")
    print(agent.print_policy(policy))

    # Save the learned policy
    policy_path = os.path.join(exp_dir, "learned_policy.txt")
    save_policy(policy, policy_path)

    # Evaluate the policy using the evaluator
    results = evaluate_policy(env, policy, num_episodes=eval_episodes)
    
    print(f"\n🏆 Mean return per episode: {results['mean_return']:.2f}")
    print(f"🎯 Success rate: {results['success_rate']:.2%}")
    print(f"⏱️ Mean steps per episode: {results['mean_steps']:.2f}")
    print(f"⚡ Training time: {training_time:.2f} seconds")

    # Save metrics 
    metrics = {
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
    
    save_metrics("metrics.csv", metrics, exp_dir)

    # Generate and save rewards plot 
    plot_path = os.path.join(exp_dir, "rewards_plot.png")
    draw_rewards(results['rewards'], plot_path)

def clear_files():
    experiments_dir = "experiments"
    if os.path.exists(experiments_dir):
        for algo_dir in ["valueIteration", "directEstimation", "qlearning", "reinforce"]:
            algo_path = os.path.join(experiments_dir, algo_dir)
            if os.path.exists(algo_path):
                for exp_dir in os.listdir(algo_path):
                    exp_path = os.path.join(algo_path, exp_dir)
                    if os.path.isdir(exp_path):
                        for file in os.listdir(exp_path):
                            os.remove(os.path.join(exp_path, file))
                        os.rmdir(exp_path)
                os.rmdir(algo_path)
        print("🧹 All experiment files have been cleared!")
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

if __name__ == "__main__":
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
