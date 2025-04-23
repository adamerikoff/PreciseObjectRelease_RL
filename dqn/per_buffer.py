import numpy as np
import random
from typing import Any, Tuple, List
from dqn.buffer import Buffer

class SumTree:
    """Helper class for the prioritized experience replay buffer that implements a sum tree."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_levels = int(np.ceil(np.log2(capacity))) + 1
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.cursor = 0

    def add(self, priority: float, data: Any) -> None:
        """Add an experience with a given priority to the tree."""
        idx = self.cursor + self.capacity - 1  # Start at leaf node
        
        self.data[self.cursor] = data
        self.update(idx, priority)
        
        self.cursor += 1
        if self.cursor >= self.capacity:
            self.cursor = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, priority: float) -> None:
        """Update the priority of a specific experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority changes up the tree."""
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get(self, value: float) -> Tuple[int, float, Any]:
        """Get the experience with priority corresponding to the given value."""
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            
            if left >= len(self.tree):
                break
                
            if value <= self.tree[left] or right >= len(self.tree):
                parent = left
            else:
                value -= self.tree[left]
                parent = right
                
        data_idx = parent - self.capacity + 1
        return parent, self.tree[parent], self.data[data_idx]

    def total_priority(self) -> float:
        """Return the total sum of priorities."""
        return self.tree[0]


class PrioritizedReplayBuffer(Buffer):
    """Prioritized Experience Replay buffer using SumTree for efficient sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences
            alpha: How much prioritization to use (0 = no prioritization)
            beta: Initial importance sampling weight (anneals to 1)
            beta_increment: How much to increment beta each sampling
            epsilon: Small constant to ensure no experience has 0 priority
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial priority for new experiences
        self.tree = SumTree(capacity)

    def add(self, experience: Any) -> None:
        """Add an experience to the buffer with the current maximum priority."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences with probabilities based on their priorities.
        
        Returns:
            Tuple of (experiences, indices, weights) where weights are for importance sampling
        """
        experiences = []
        indices = []
        priorities = []
        weights = np.empty(batch_size, dtype=np.float32)
        
        segment = self.tree.total_priority() / batch_size
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, exp = self.tree.get(value)
            
            priorities.append(priority)
            experiences.append(exp)
            indices.append(idx)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total_priority()
        weights = (self.size * sampling_probabilities) ** -self.beta
        weights /= weights.max()  # Normalize for stability
        
        # Update beta (importance sampling exponent)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update the priorities of the experiences at the given indices."""
        priorities = np.array(priorities) + self.epsilon  # Ensure no priority is zero
        priorities = np.minimum(priorities, self.max_priority)  # Clip if necessary
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority ** self.alpha)
        
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        """Return the current number of stored experiences."""
        return self.size