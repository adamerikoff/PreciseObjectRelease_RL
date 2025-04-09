import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Expanded architecture for 3D environment
        self.fc1 = nn.Linear(state_size, 256)  # First hidden layer (increased)
        self.fc2 = nn.Linear(256, 128)         # Second hidden layer (increased)
        self.fc3 = nn.Linear(128, 64)          # Third hidden layer (increased)
        self.fc4 = nn.Linear(64, 32)           # Additional hidden layer
        self.fc5 = nn.Linear(32, action_size)  # Output layer
        
    def forward(self, state):
        # Forward pass with ReLU activations
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)  # Output Q-values

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state,        # np.ndarray (state_size,)
            action,       # int
            reward,       # float
            next_state,   # np.ndarray (state_size,)
            done          # bool
        ))

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)

        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        dones = torch.FloatTensor([float(x[4]) for x in batch]).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns current number of stored experiences"""
        return len(self.buffer)
    
    def to_dataframe(self):
        """Convert buffer contents to a pandas DataFrame"""
        data = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': []
        }
        
        for experience in self.buffer:
            state, action, reward, next_state, done = experience
            data['state'].append(state.copy())
            data['action'].append(action)
            data['reward'].append(reward)
            data['next_state'].append(next_state.copy())
            data['done'].append(done)
        
        return pd.DataFrame(data)
    
    def dump_to_csv(self, filename):
        """Dump buffer contents to a CSV file"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 buffer_size=10000, batch_size=128, gamma=0.99, lr=5e-4,
                 tau=0.005, update_every=4, device=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): size of replay buffer
            batch_size (int): size of each training batch
            gamma (float): discount factor
            lr (float): learning rate
            tau (float): interpolation parameter for soft update of target network
            update_every (int): how often to update the network
            device (torch.device): device to use for tensors (cpu or cuda)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                if experiences is not None:
                    self.learn(experiences, self.gamma)

    def act(self, state, eps):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon for epsilon-greedy action selection
        """
        state = torch.from_numpy( np.array(state, dtype=np.float32)).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename):
        """Save the local Q-network parameters."""
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        """Load the local Q-network parameters."""
        self.qnetwork_local.load_state_dict(torch.load(filename))