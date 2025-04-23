from typing import Any, Tuple, List
import random

from dqn.buffer import Buffer

class ReplayBuffer(Buffer):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer = []

    def add(self, experience: Any) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Any], None]:
        samples = random.sample(self.buffer, batch_size)
        return samples, None
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        pass