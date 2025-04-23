from dqn.agent import DQN
from dqn.q_network import QNetworkShallow, QNetworkMedium, QNetworkDeep, QNetworkTief
from dqn.vanilla_buffer import ReplayBuffer

from training.training_template import train, train_jump

def train_vanilla_dqn():
    buffer = ReplayBuffer
    network = QNetworkDeep
    train("Vanilla DQN", network, buffer, "dqn_vanilla", render=False)

def train_vanilla_dqn_jump():
    buffer = ReplayBuffer
    network = QNetworkDeep
    train_jump("VanillaJUMP DQN", network, buffer, "dqn_vanilla_jump", render=False)

train_vanilla_dqn_jump()
train_vanilla_dqn()
