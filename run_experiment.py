import gymnasium as gym
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.value_iteration import ValueIterationAgent  # tu fichero de agente separado

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
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    pd.DataFrame(data).to_csv(path, index=False)
    return path

def run_value_iteration_experiment():

    print("====================")
    print("  Value Iteration  ")
    print("====================")

    gamma = float(input("Introduce el valor de gamma (factor de descuento) [ej. 0.95]: "))
    num_episodes = int(input("Número de episodios para evaluación [ej. 100]: "))

    # Crear entorno y agente
    env = gym.make("CliffWalking-v0", render_mode="ansi", is_slippery=True)
    agent = ValueIterationAgent(env, gamma=gamma)

    rewards, max_diffs = agent.train()

    policy = agent.policy()
    print("\nPolítica aprendida:")
    print(agent.print_policy(policy))
    avg_reward = evaluate_policy(env, policy, num_episodes=num_episodes)

    print(f"\n🎯 Recompensa media tras entrenamiento: {avg_reward:.2f}")

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"value_iteration_results_{timestamp}.csv"
    save_path = save_results(filename, {
        "iteration": list(range(1, len(max_diffs)+1)),
        "max_diff": max_diffs
    })
    print(f"📁 Resultados guardados en: {save_path}")

if __name__ == "__main__":
    print("\n")
    print("🤖 === Cliff Walking Experiment Runner === 🎮\n")
    run_value_iteration_experiment()
