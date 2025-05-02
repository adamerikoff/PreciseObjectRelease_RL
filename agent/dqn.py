import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = np.vstack([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.vstack([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_size)
        )
    
    def forward(self, state):
        return self.model(state)

class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size,
        gamma=0.99,
        learning_rate=0.001,
        buffer_size=100000,
        batch_size=64,
        update_interval=100,
        tau=0.005,
        device="cpu"
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.tau = tau

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.device = device

        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

        self.time_step = 0

    def act(self, state, epsilon=0.01):
        if random.random() > epsilon:
            state_tensor = torch.tensor(state).float().to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state_tensor)
            self.q_network.train()
            return torch.argmax(action_values).item()
        else:
            return random.choice(range(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.time_step += 1

        if self.time_step % self.update_interval == 0 and len(self.memory) > self.batch_size:
            self.learn()
            self.update_target_network()

    def learn(self):
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device).unsqueeze(1)

        q_next_target = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (self.gamma * q_next_target * (1 - dones))

        q_current = self.q_network(states).gather(1, actions)

        loss = nn.MSELoss()(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_network()