import pandas as pd
import matplotlib.pyplot as plt

def plot_training_progress(df):
    """Helper function to plot training progress"""
    plt.figure(figsize=(12, 6))
    
    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(df['episode'], df['avg100'], label='100-episode avg')
    plt.plot(df['episode'], df['reward'], alpha=0.3, label='Episode reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Epsilon plot
    plt.subplot(1, 2, 2)
    plt.plot(df['episode'], df['epsilon'])
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

training_stats = pd.read_csv('training_stats_20250409_204928.csv')

plot_training_progress(training_stats)