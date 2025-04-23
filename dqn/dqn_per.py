import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.buffer import Buffer

class DQN_PER:
    def __init__(
        self,
        q_network: nn.Module,
        buffer: Buffer,
        action_size: int,
        gamma: float = 0.99,
        lr: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        tau: float = 0.01,
        device: str = "cpu"
    ):
        """
        Args:
            q_network: Q-network (any of your QNetwork* classes)
            buffer: Replay buffer instance
            action_size: Number of possible actions
            gamma: Discount factor
            lr: Learning rate
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            tau: Soft update coefficient for target network
            device: "cpu" or "cuda"
        """
        self.q_network = q_network.to(device)
        self.target_network = type(q_network)().to(device)  # Create copy
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.buffer = buffer
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.device = device
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self, batch_size: int) -> Optional[float]:
        """Train on a batch from prioritized replay buffer"""
        if len(self.buffer) < batch_size:
            return None
        
        # Sample batch - now returns (experiences, indices, weights)
        samples, indices, weights = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(np.array(weights)).to(self.device)
        
        # Current Q values for taken actions
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute TD errors (for priority updates)
        td_errors = (target_q - current_q.squeeze()).abs().detach().numpy()
        
        # Compute weighted loss
        loss = (weights * self.loss_fn(current_q.squeeze(), target_q)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_errors)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Soft update target network
        self._soft_update_target_network()
        
        return loss.item()

    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']