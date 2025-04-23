from plot.plotting import plot_training_stats_from_csv

plot_training_stats_from_csv('csv_data/dqn_vanilla.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla")
plot_training_stats_from_csv('csv_data/dqn_vanilla_jump.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla_jump")