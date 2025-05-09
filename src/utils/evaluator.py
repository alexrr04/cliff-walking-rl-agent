import time
import pandas as pd
import os

def rollout(env, policy, max_steps=300):
    """
    Execute one episode with the greedy policy.

    Args:
        env: Gymnasium environment instance
        policy: Array of actions for each state
        max_steps: Maximum number of steps per episode

    Returns:
        tuple: (reached_goal: bool, steps: int, total_return: float)
    """
    state, _ = env.reset()
    total_return = 0.0
    for t in range(1, max_steps + 1):
        action = policy[state]
        state, reward, is_done, truncated, _ = env.step(action)
        total_return += reward

        if is_done:  # reached [3,11]
            print(f"\n🎉 Goal reached in {t} steps, return = {total_return}\n")
            return True, t, total_return
        if truncated:  # hit the TimeLimit wrapper
            break

    print("\n💥 Episode ended without reaching the goal\n")
    return False, t, total_return

def evaluate_policy(env, policy, num_episodes=100, max_steps=200, algo_dir=None):
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: Gymnasium environment instance
        policy: Array of actions for each state
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        algo_dir: Name of the algorithm directory (e.g., "valueIteration", "directEstimation")

    Returns:
        dict: Evaluation metrics including success rate, mean steps, mean return, and episode details
    """
    successes = 0
    total_steps = 0
    total_reward = 0
    test_rewards = []
    episode_metrics = []

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep + 1} ===")
        start_time = time.time()
        reached_goal, steps, episode_return = rollout(env, policy, max_steps=max_steps)
        episode_time = time.time() - start_time
        
        # Store episode metrics
        episode_metrics.append({
            'episode': ep,
            'reward': episode_return,
            'steps': steps,
            'time': episode_time,
            'reached_goal': reached_goal
        })
        
        test_rewards.append(episode_return)
        successes += int(reached_goal)
        total_steps += steps
        total_reward += episode_return

    success_rate = successes / num_episodes
    mean_steps = total_steps / num_episodes
    mean_return = total_reward / num_episodes

    # Save episode metrics to CSV if algo_dir is provided
    if algo_dir:
        metrics_df = pd.DataFrame(episode_metrics)
        csv_path = os.path.join("experiments", algo_dir, "latest", "evaluation_metrics.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        metrics_df.to_csv(csv_path, index=False)

    return {
        'success_rate': success_rate,
        'mean_steps': mean_steps,
        'mean_return': mean_return,
        'rewards': test_rewards,
        'episode_metrics': episode_metrics
    }
