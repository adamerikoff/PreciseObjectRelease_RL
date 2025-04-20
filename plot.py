import pandas as pd
import matplotlib.pyplot as plt

def plot_training_progress(df, filename):
    """Helper function to plot training progress"""
    plt.figure(figsize=(16, 12))

    # Reward plot
    plt.subplot(2, 2, 1)
    plt.plot(df['episode'], df['avg100'], label='100-episode avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)

    # Steps plot
    plt.subplot(2, 2, 2)
    plt.plot(df['episode'], df['steps'], label='Episode steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True)

    # Success Rate plot
    plt.subplot(2, 2, 3)
    if 'avg100success' in df.columns:
        plt.plot(df['episode'], df['avg100success'], label='100-episode avg success rate (%)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate')
        plt.legend()
        plt.grid(True)
    else:
        plt.title('Success Rate Data Not Available')

    # Epsilon plot
    plt.subplot(2, 2, 4)
    plt.plot(df['episode'], df['epsilon'])
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()