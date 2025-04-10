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
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, state):
        return self.net(state)

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
        return len(self.buffer)
    
    def to_dataframe(self):
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
        df = self.to_dataframe()
        df.to_csv(filename, index=False)


class DQNAgent:
    def __init__(self, state_size, action_size,
                 buffer_size=10000, batch_size=128, gamma=0.99, lr=5e-4,
                 tau=0.005, update_every=4, device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_counter = 0


        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.target_counter = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                
                self.target_counter += 1
                self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps):
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

    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename))