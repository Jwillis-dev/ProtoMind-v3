# protomind/task/planner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any

class TaskPlannerModule(nn.Module):
    """
    Ranks or selects candidate actions/tools/goals from hypotheses.
    Now also supports returning actual candidate objects for executive use.
    """
    def __init__(self, abstraction_size: int, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(abstraction_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, candidates_tensor: torch.Tensor, candidates_list: List[Any] = None) -> Tuple[Any, torch.Tensor]:
        """
        Args:
            candidates_tensor: (batch, num_candidates, abstraction_size)
            candidates_list: optional list of candidate objects corresponding to tensor indices
        Returns:
            selected_candidate: object from candidates_list or index if not provided
            scores: tensor of shape (batch, num_candidates)
        """
        scores = self.scorer(candidates_tensor).squeeze(-1)  # (batch, num_candidates)
        selected_idx = scores.argmax(dim=-1)

        if candidates_list is not None:
            if isinstance(selected_idx, torch.Tensor) and selected_idx.dim() == 1:
                selected_candidate = [candidates_list[i][idx.item()] for i, idx in enumerate(selected_idx)]
            else:
                selected_candidate = candidates_list[selected_idx.item()]
            return selected_candidate, scores
        else:
            return selected_idx, scores
