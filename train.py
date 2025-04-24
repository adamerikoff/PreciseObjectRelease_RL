from training import training_base
from plot.plotting import plot_training_stats_from_csv

training_base.train("Vanilla_DQN_JUMP", "dqn_vanilla_jump", render=False, jump=True, top=True)
training_base.train("Vanilla_DQN", "dqn_vanilla", render=False, jump=False, top=True)

plot_training_stats_from_csv('csv_data/dqn_vanilla.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla")
plot_training_stats_from_csv('csv_data/dqn_vanilla_jump.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla_jump")
