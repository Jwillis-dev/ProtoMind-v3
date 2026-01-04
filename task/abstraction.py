# protomind/core/task_abstraction.py
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TaskAbstractionModule(nn.Module):
    """
    Converts raw observations or task descriptions into structured
    abstract representations (goal, constraints, possible actions).
    """
    def __init__(self, input_size: int, hidden_size: int, abstraction_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.abstraction_size = abstraction_size

        # Encode raw input into hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Project into abstract task space
        self.goal_head = nn.Linear(hidden_size, abstraction_size)
        self.constraint_head = nn.Linear(hidden_size, abstraction_size)
        self.action_head = nn.Linear(hidden_size, abstraction_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: raw input tensor (batch_size, input_size)
        Returns:
            dict: {
                'goal': tensor (batch_size, abstraction_size),
                'constraints': tensor (batch_size, abstraction_size),
                'potential_actions': tensor (batch_size, abstraction_size)
            }
        """
        h = self.encoder(x)

        goal = self.goal_head(h)
        constraints = self.constraint_head(h)
        potential_actions = self.action_head(h)

        return {
            'goal': goal,
            'constraints': constraints,
            'potential_actions': potential_actions
        }

    def sample_goal(self, x: torch.Tensor):
        """
        Optionally sample a discrete goal from continuous embedding.
        Could be used for discrete simulation environments.
        """
        abstraction = self.forward(x)
        goal = F.softmax(abstraction['goal'], dim=-1)
        return goal.argmax(dim=-1)

