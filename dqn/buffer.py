from abc import ABC, abstractmethod
from typing import Any, Tuple, List

class Buffer(ABC):
    def __init__(self, capacity: int):
        """
        Initialize the buffer with a maximum capacity.
        
        Args:
            capacity: Maximum number of experiences the buffer can hold
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0
    
    @abstractmethod
    def add(self, experience: Any) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: The experience to be added (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[Any, Any]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple containing the sampled experiences and their indices/weights
        """
        pass
    
    @abstractmethod
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for specific experiences (if applicable).
        
        Args:
            indices: List of experience indices to update
            priorities: New priorities for these experiences
        """
        pass
    
    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        """
        return self.size