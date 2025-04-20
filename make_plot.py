import pandas as pd
import matplotlib.pyplot as plt
import os

def create_comparison_plots(df1, df2, label1="Standard", label2="Modified", output_dir="plots"):
    """Create 4 separate comparison plots for two datasets"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set consistent style for all plots (updated style name)
    plt.style.use('seaborn-v0_8')  # Modern equivalent of 'seaborn'
    comparison_colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    # 1. Reward Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1['avg100'], label=f'{label1} (100-ep avg)', color=comparison_colors[0], linewidth=2)
    plt.plot(df2['episode'], df2['avg100'], label=f'{label2} (100-ep avg)', color=comparison_colors[1], linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Reward Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Steps Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1['steps'], label=label1, color=comparison_colors[0], alpha=0.7)
    plt.plot(df2['episode'], df2['steps'], label=label2, color=comparison_colors[1], alpha=0.7)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    plt.title('Episode Length Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/steps_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Success Rate Comparison Plot (if available)
    if 'avg100success' in df1.columns and 'avg100success' in df2.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df1['episode'], df1['avg100success'], label=f'{label1} (100-ep avg)', 
                color=comparison_colors[0], linewidth=2)
        plt.plot(df2['episode'], df2['avg100success'], label=f'{label2} (100-ep avg)', 
                color=comparison_colors[1], linewidth=2)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Success Rate Comparison', fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Epsilon Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1['epsilon'], label=label1, color=comparison_colors[0])
    plt.plot(df2['episode'], df2['epsilon'], label=label2, color=comparison_colors[1])
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Epsilon', fontsize=12)
    plt.title('Exploration Rate Comparison', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/epsilon_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
if __name__ == "__main__":
    # Load your CSV files
    df_modified = pd.read_csv('training_stats_dqn_jump.csv')
    df_standard = pd.read_csv('training_stats_dqn_vanilla.csv')
    
    # Generate comparison plots
    create_comparison_plots(df_standard, df_modified, 
                          label1="Standard Replay", 
                          label2="Modified Replay")