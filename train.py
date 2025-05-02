from training import train_base
# from plot.plotting import plot_training_stats_from_csv

train_base.train("DQN_JUMP", "dqn_jump", render=False, jump=True, top=True)
# train_base.train("DQN", "dqn", render=False, jump=False, top=True)


# plot_training_stats_from_csv('csv_data/dqn_vanilla.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla")
# plot_training_stats_from_csv('csv_data/dqn_vanilla_jump.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla_jump")