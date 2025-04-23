from dqn.agent import DQN
from dqn.q_network import QNetworkShallow, QNetworkMedium, QNetworkDeep, QNetworkTief
from dqn.vanilla_buffer import ReplayBuffer

from training.training_template import train, train_jump
from plot.plotting import plot_training_stats_from_csv

def train_vanilla_dqn():
    buffer = ReplayBuffer
    network = QNetworkTief
    train("Vanilla DQN", network, buffer, "dqn_vanilla", render=False)

def train_vanilla_dqn_jump():
    buffer = ReplayBuffer
    network = QNetworkTief
    train_jump("VanillaJUMP DQN", network, buffer, "dqn_vanilla_jump", render=False)

train_vanilla_dqn_jump()
train_vanilla_dqn()

plot_training_stats_from_csv('csv_data/dqn_vanilla.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla")
plot_training_stats_from_csv('csv_data/dqn_vanilla_jump.csv', save_plots=True, show_plots=False, output_dir="plots/dqn_vanilla_jump")