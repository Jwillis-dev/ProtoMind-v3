import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class Experience:
    """Stores a single step of agent experience."""
    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    done: bool
    internal_state: Optional[Dict[str, Any]] = None


class AttentionMemory(nn.Module):
    """
    Simple attention-based working memory.
    Stores a fixed number of memory slots and attends to them.
    """
    def __init__(self, hidden_size: int, memory_slots: int, device: str = 'cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.device = device
        
        # Memory matrix: [memory_slots, hidden_size]
        self.memory = torch.zeros(memory_slots, hidden_size, device=device)
        self.write_head = 0
        
        # Attention mechanism
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def attend(self, h: torch.Tensor):
        """
        Attend to memory given current hidden state h.
        Args:
            h: [batch_size, hidden_size]
        Returns:
            attended: [batch_size, hidden_size]
            weights: [batch_size, memory_slots]
        """
        batch_size = h.shape[0]
        
        # Linear projections
        q = self.query(h)                   # [batch_size, hidden_size]
        k = self.key(self.memory)           # [memory_slots, hidden_size]
        v = self.value(self.memory)         # [memory_slots, hidden_size]
        
        # Attention scores & weights
        scores = torch.matmul(q, k.transpose(0, 1))  # [batch_size, memory_slots]
        weights = torch.softmax(scores, dim=-1)
        
        # Attended memory
        attended = torch.matmul(weights, v)          # [batch_size, hidden_size]
        return attended, weights

    def update(self, h: torch.Tensor):
        """
        Update memory with new hidden state.
        Uses a simple circular buffer.
        """
        self.memory[self.write_head] = h.mean(dim=0).detach()
        self.write_head = (self.write_head + 1) % self.memory_slots


class MemoryStream:
    """
    Optional storage of experiences for inspection.
    Not used by ProtoMind internally.
    """
    def __init__(self, max_size: int = 10000):
        self.experiences: List[Experience] = []
        self.max_size = max_size

    def add(self, experience: Experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def get_recent(self, n: int = 100) -> List[Experience]:
        return self.experiences[-n:]

    def get_all(self) -> List[Experience]:
        return self.experiences
