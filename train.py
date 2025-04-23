from dqn.q_network import QNetworkTief

from dqn.vanilla_buffer import ReplayBuffer
from dqn.per_buffer import PrioritizedReplayBuffer

from training import training_base
from training import training_per
from plot.plotting import plot_training_stats_from_csv

def train_vanilla_dqn():
    buffer = ReplayBuffer
    network = QNetworkTief
    training_base.train("Vanilla DQN", network, buffer, "dqn_vanilla", render=False)

def train_vanilla_dqn_jump():
    buffer = ReplayBuffer
    network = QNetworkTief
    training_base.train_jump("VanillaJUMP DQN", network, buffer, "dqn_vanilla_jump", render=False)

def train_vanilla_dqn_per():
    buffer = PrioritizedReplayBuffer
    network = QNetworkTief
    training_per.train("PER DQN", network, "dqn_per", render=False)

def train_vanilla_dqn_jump_per():
    buffer = PrioritizedReplayBuffer
    network = QNetworkTief
    training_per.train_jump("PERJUMP DQN", network, "dqn_per_jump", render=False)

train_vanilla_dqn_jump()
train_vanilla_dqn()

train_vanilla_dqn_per()
train_vanilla_dqn_jump_per()

plot_training_stats_from_csv('csv_data/dqn_vanilla.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla")
plot_training_stats_from_csv('csv_data/dqn_vanilla_jump.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla_jump")

plot_training_stats_from_csv('csv_data/dqn_per.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_per")
plot_training_stats_from_csv('csv_data/dqn_per_jump.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_per_jump")
