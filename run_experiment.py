import time
import os
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

# Importa algoritmos desde src/
from src.value_iteration import run_value_iteration
from src.direct_estimation import run_direct_estimation
from src.qlearning import run_q_learning
from src.reinforce import run_reinforce

ALGORITHMS = {
    "1": ("Value Iteration", run_value_iteration),
    "2": ("Monte Carlo (Direct Estimation)", run_direct_estimation),
    "3": ("Q-Learning", run_q_learning),
    "4": ("REINFORCE", run_reinforce),
}

def ask_float(prompt, default):
    try:
        return float(input(f"{prompt} (default={default}): ") or default)
    except ValueError:
        return default

def ask_int(prompt, default):
    try:
        return int(input(f"{prompt} (default={default}): ") or default)
    except ValueError:
        return default

def get_parameters(algo_name):
    params = {}
    params['episodes'] = ask_int("Número de episodios", 1000)
    params['gamma'] = ask_float("Factor de descuento (gamma)", 0.99)

    if algo_name == "Q-Learning":
        params['alpha'] = ask_float("Tasa de aprendizaje (alpha)", 0.1)
        params['epsilon'] = ask_float("Epsilon (exploración)", 1.0)
        params['epsilon_decay'] = ask_float("Decay de epsilon", 0.995)
        params['min_epsilon'] = ask_float("Mínimo epsilon", 0.05)

    if algo_name == "REINFORCE":
        params['learning_rate'] = ask_float("Learning rate", 0.1)
        params['lr_decay'] = ask_float("Learning rate decay", 0.99)

    return params

def plot_results(rewards, title, save_path=None):
    plt.figure(figsize=(10, 5))
    rolling = pd.Series(rewards).rolling(50).mean()
    plt.plot(rewards, label="Recompensa")
    plt.plot(rolling, label="Media móvil", color="red")
    plt.title(title)
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_experiment_results(base_path, algo_name, config, results):
    os.makedirs(base_path, exist_ok=True)

    np.savetxt(os.path.join(base_path, "rewards.csv"), results["rewards"], delimiter=",")
    
    if "policy" in results and results["policy"] is not None:
        np.savetxt(os.path.join(base_path, "policy.txt"), results["policy"], fmt="%d")

    plot_results(results["rewards"], f"{algo_name} - Recompensa", os.path.join(base_path, "reward_plot.png"))

    with open(os.path.join(base_path, "config.log"), "w") as f:
        f.write(f"Algoritmo: {algo_name}\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

def main():
    print("=== Experiment Runner ===\n")
    for k, (name, _) in ALGORITHMS.items():
        print(f"{k}. {name}")
    choice = input("\nElige un algoritmo [1-4]: ").strip()
    algo_name, algo_fn = ALGORITHMS.get(choice, (None, None))

    if not algo_fn:
        print("❌ Opción inválida.")
        return

    print(f"\nElegido: {algo_name}")
    config = get_parameters(algo_name)

    env = gym.make("CliffWalking-v0", is_slippery=True)
    print("\n🚀 Entrenando...")
    start = time.time()
    results = algo_fn(env, config)
    duration = time.time() - start
    print(f"\n✅ Entrenamiento completado en {duration:.2f} segundos")

    timestamp = int(time.time())
    base_path = os.path.join("results", f"{algo_name.replace(' ', '_').lower()}_{timestamp}")
    save_experiment_results(base_path, algo_name, config, results)

    print(f"📁 Resultados guardados en: {base_path}")

if __name__ == "__main__":
    main()
