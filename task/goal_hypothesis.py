# protomind/core/goal_hypothesis.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GoalHypothesisModule(nn.Module):
    """
    Generates candidate goal hypotheses from task abstraction.
    """
    def __init__(self, abstraction_size: int, hidden_size: int, num_hypotheses: int = 5):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        self.hidden_size = hidden_size

        self.generator = nn.Sequential(
            nn.Linear(abstraction_size * 3, hidden_size),  # goal + constraints + potential actions
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.hypothesis_head = nn.Linear(hidden_size, abstraction_size * num_hypotheses)

    def forward(self, task_abstraction: dict):
        """
        Args:
            task_abstraction: dict from TaskAbstractionModule
        Returns:
            tensor (batch_size, num_hypotheses, abstraction_size)
        """
        combined = torch.cat([
            task_abstraction['goal'],
            task_abstraction['constraints'],
            task_abstraction['potential_actions']
        ], dim=-1)
        
        h = self.generator(combined)
        hypotheses = self.hypothesis_head(h)
        hypotheses = hypotheses.view(-1, self.num_hypotheses, task_abstraction['goal'].shape[-1])
        return hypotheses
