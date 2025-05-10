import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import pandas as pd
import numpy as np
import glob
import argparse
import time
import yaml
import shutil
from datetime import datetime
from termcolor import colored
from itertools import product

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

def get_input_with_default(prompt, default, type_cast=float):
    """Get input from user with a default value if no input is provided"""
    value = input(f"{prompt} [default: {default}]: ").strip()
    return type_cast(value) if value else type_cast(default)

def run_value_iteration_experiment(exp_dir, params=None):
    print("\n--------------------------")
    print("  🤖 Value Iteration 🤖 ")
    print("--------------------------\n")

    # Use parameters from config if provided, otherwise get from user input
    if params is not None:
        gamma = params.get('gamma', 0.95)
        num_episodes = params.get('num_episodes', 500)
        epsilon = params.get('epsilon', 0.00000001)
    else:
        gamma = get_input_with_default("Enter gamma value (discount factor)", 0.95)
        num_episodes = get_input_with_default("Number of episodes for evaluation", 500, int)
        epsilon = get_input_with_default("Enter epsilon value for convergence", 0.00000001)

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


def run_direct_estimation_experiment(exp_dir, params=None):
    print("\n---------------------------")
    print("  🤖 Direct Estimation 🤖 ")
    print("----------------------------\n")

    # Use parameters from config if provided, otherwise get from user input
    if params is not None:
        gamma = params.get('gamma', 0.95)
        num_episodes = params.get('num_episodes', 500)
        num_trajectories = params.get('num_trajectories', 500)
        max_iters = params.get('max_iters', 500)
        patience = params.get('patience', 100)
        num_runs = params.get('num_runs', 5)
    else:
        gamma = get_input_with_default("Enter gamma value (discount factor)", 0.95)
        num_episodes = get_input_with_default("Number of episodes for evaluation", 500, int)
        num_trajectories = get_input_with_default("Number of trajectories for sampling", 500, int)
        max_iters = get_input_with_default("Maximum iterations for training", 500, int)
        patience = get_input_with_default("Patience for convergence", 100, int)
        num_runs = get_input_with_default("Number of runs to execute", 5, int)

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

def run_qlearning_experiment(exp_dir, params=None):
    print("\n--------------------------")
    print("  🤖 Q-Learning 🤖 ")
    print("--------------------------\n")

    # Use parameters from config if provided, otherwise get from user input
    if params is not None:
        gamma = params.get('gamma', 0.95)
        num_episodes = params.get('num_episodes', 1000)
        learning_rate = params.get('learning_rate', 0.1)
        epsilon = params.get('epsilon', 0.9)
        ep_decay = params.get('ep_decay', 0.95)
        lr_decay = params.get('lr_decay', 0.95)
        eval_episodes = params.get('eval_episodes', 500)
        penalty = params.get('penalty', -1.0)
        num_runs = params.get('num_runs', 5)
    else:
        gamma = get_input_with_default("Enter gamma value (discount factor)", 0.95)
        num_episodes = get_input_with_default("Number of episodes for training", 1000, int)
        learning_rate = get_input_with_default("Enter learning rate", 0.1)
        epsilon = get_input_with_default("Enter initial epsilon value", 0.9)
        ep_decay = get_input_with_default("Enter epsilon decay rate", 0.95)
        lr_decay = get_input_with_default("Enter learning rate decay rate", 0.95)
        eval_episodes = get_input_with_default("Number of episodes for evaluation", 500, int)
        penalty = get_input_with_default("Enter penalty for moving to the right", -1.0)
        num_runs = get_input_with_default("Number of runs to execute", 5, int)

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
        agent = QLearningAgent(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon, ep_decay=ep_decay, lr_decay=lr_decay)

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
            "epsilon_decay": ep_decay,
            "learning_rate_decay": lr_decay,
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

def run_reinforce_experiment(exp_dir, params=None):
    print("\n--------------------------")
    print("  🤖 REINFORCE 🤖 ")
    print("--------------------------\n")

    # Use parameters from config if provided, otherwise get from user input
    if params is not None:
        gamma = params.get('gamma', 0.9)
        learning_rate = params.get('learning_rate', 0.99)
        lr_decay = params.get('lr_decay', 0.99)
        training_episodes = params.get('training_episodes', 1000)
        t_max = params.get('t_max', 200)
        eval_episodes = params.get('eval_episodes', 100)
        num_runs = params.get('num_runs', 5)
    else:
        gamma = get_input_with_default("Enter gamma value (discount factor)", 0.9)
        learning_rate = get_input_with_default("Enter learning rate", 0.99)
        lr_decay = get_input_with_default("Enter learning rate decay", 0.99)
        training_episodes = get_input_with_default("Number of episodes for training", 1000, int)
        t_max = get_input_with_default("Enter maximum steps per episode", 200, int)
        eval_episodes = get_input_with_default("Number of episodes for evaluation", 100, int)
        num_runs = get_input_with_default("Number of runs to execute", 5, int)

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
    if not os.path.exists(experiments_dir):
        print("📂 No experiment files found.")
        return

    for algo_dir in ["valueIteration", "directEstimation", "qlearning", "reinforce"]:
        algo_path = os.path.join(experiments_dir, algo_dir)
        if os.path.exists(algo_path):
            for item in os.listdir(algo_path):
                if item.startswith("experiment_"):
                    item_path = os.path.join(algo_path, item)
                    if os.path.isdir(item_path):
                        # Use shutil.rmtree to recursively remove directory and contents
                        shutil.rmtree(item_path)
            
            # Remove algorithm directory if empty
            if not os.listdir(algo_path):
                os.rmdir(algo_path)
    
    print("🧹 Default experiment files have been cleared!")

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

def load_experiment_config(config_path):
    """Load and validate experiment configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['algorithm', 'parameters']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file")
    
    # Validate algorithm name
    valid_algorithms = {
        'valueIteration': 1,
        'directEstimation': 2,
        'qlearning': 3,
        'reinforce': 4
    }
    if config['algorithm'] not in valid_algorithms:
        raise ValueError(f"Invalid algorithm. Must be one of: {', '.join(valid_algorithms.keys())}")
    
    return config

def generate_parameter_combinations(parameters):
    """Generate all parameter combinations from config"""
    # Convert list parameters to lists if they're not already
    param_lists = {
        k: v if isinstance(v, list) else [v]
        for k, v in parameters.items()
    }
    
    # Generate all combinations
    keys = param_lists.keys()
    values = param_lists.values()
    combinations = list(product(*values))
    
    # Convert combinations to list of dicts
    return [dict(zip(keys, combo)) for combo in combinations]

def print_combination_header(index, total, params, start_time):
    """Print a visually distinct header for each parameter combination"""
    border_top = "╔" + "═" * 53 + "╗"
    border_mid = "╠" + "═" * 53 + "╣"
    border_bot = "╚" + "═" * 53 + "╝"
    
    print("\n" + border_top)
    print(f"║ {colored(f'Parameter Combination {index + 1}/{total}', 'cyan', attrs=['bold']): <52}║")
    print(f"║ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S'): <42}║")
    print(border_mid)
    print(f"║ {colored('Parameters:', 'yellow', attrs=['bold']): <52}║")
    
    # Print parameters in a formatted table
    for k, v in params.items():
        key = colored(f"{k}:", 'green')
        print(f"║ {key: <20} {v: <32}║")
    
    print(border_bot)

def print_combination_footer(start_time):
    """Print a footer with timing information"""
    duration = time.time() - start_time
    border = "═" * 54
    print(f"\n⏱️  {colored('Combination completed in:', 'cyan')} {duration:.2f}s")
    print(border)

def run_experiment_from_config(config_path):
    """Run experiments using configuration from YAML file"""
    print(f"\n📄 Loading configuration from {config_path}...")
    config = load_experiment_config(config_path)
    
    # Map algorithm name to ID and function
    algo_map = {
        'valueIteration': (1, run_value_iteration_experiment),
        'directEstimation': (2, run_direct_estimation_experiment),
        'qlearning': (3, run_qlearning_experiment),
        'reinforce': (4, run_reinforce_experiment)
    }
    
    algorithm_id, run_func = algo_map[config['algorithm']]
    combinations = generate_parameter_combinations(config['parameters'])
    
    print(colored(f"\n🔬 Running {len(combinations)} experiment combinations for {config['algorithm']}", 'cyan', attrs=['bold']))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("experiments", config['algorithm'], f"experiment_{timestamp}")
    
    total_start_time = time.time()
    
    for i, params in enumerate(combinations):
        # Create experiment directory for this combination
        exp_dir = os.path.join(base_dir, f"combination_{i}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Set up logging
        log_filename = os.path.join(exp_dir, "experiment_log.txt")
        sys.stdout = Logger(log_filename)
        
        combination_start_time = time.time()
        print_combination_header(i, len(combinations), params, datetime.now())
        
        # Run experiment with parameters
        run_func(exp_dir, params)
        
        print_combination_footer(combination_start_time)
    
    # Restore original stdout
    sys.stdout = sys.__stdout__
    total_duration = time.time() - total_start_time
    print(colored(f"\n✅ All experiment combinations completed in {total_duration:.2f}s", 'green', attrs=['bold']))
    print(colored(f"📂 Results saved in: {base_dir}", 'cyan'))

def main():
    parser = argparse.ArgumentParser(description="🤖 Cliff Walking Experiment Runner")
    parser.add_argument('--clean', action='store_true', help='Clear all experiment files')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()

    if args.clean:
        clear_files()
    elif args.config:
        run_experiment_from_config(args.config)
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
