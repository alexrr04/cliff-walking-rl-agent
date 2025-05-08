import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_rewards(rewards, save_path):
    """
    Plot and save the rewards obtained during testing.

    Args:
        rewards (list): List of rewards to plot
        save_path (str): Path where to save the plot
    """
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
