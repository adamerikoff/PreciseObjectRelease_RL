import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import colorsys

def get_random_dark_color():
    """Generates a random dark color in hexadecimal format."""
    h = random.random()
    s = random.uniform(0.6, 0.9)  # High saturation for vibrant colors
    v = random.uniform(0.6, 0.9)  # Lower value for darker shades
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

def plot_training_stats_from_csv(csv_path, save_plots=True, show_plots=False, output_dir='plots', format='png'):
    """
    Read training statistics from CSV and plot each metric in separate plots.

    Args:
        csv_path (str): Path to the CSV file
        save_plots (bool): Whether to save plots to files (default True)
        show_plots (bool): Whether to display plots (default False)
        output_dir (str): Directory to save plots (default 'plots')
        format (str): Image format ('png', 'jpg', 'svg', 'pdf', etc.)
    """
    try:
        # Read the CSV file
        training_stats = pd.read_csv(csv_path)

        if len(training_stats) == 0:
            print("No data to plot")
            return

        # Create output directory if it doesn't exist
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the filename for the plot title
        filename = os.path.splitext(os.path.basename(csv_path))[0]

        # Get all columns except 'episode' (x-axis)
        metrics = [col for col in training_stats.columns if col != 'episode']

        color = get_random_dark_color()

        for metric in metrics:
            plt.figure(figsize=(10, 6))    
            plt.plot(training_stats['episode'], training_stats[metric], color=color)
            plt.title(f"{filename.replace('_', ' ').title()} - {metric.replace('_', ' ').title()}")
            plt.xlabel('Episode')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)

            # Save individual plots if requested
            if save_plots:
                output_path = os.path.join(output_dir, f"{filename}_{metric}.{format}")
                plt.savefig(output_path, bbox_inches='tight', format=format)
                print(f"Saved {output_path}")

            # Show individual plots if requested
            if show_plots:
                plt.tight_layout()
                plt.show()

            plt.close()  # Close the current figure to free memory

    except Exception as e:
        print(f"Error: {str(e)}")