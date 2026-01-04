import torch
import torch.nn as nn

class GenerativeModelModule(nn.Module):
    """A module for predicting the next observation."""
    def __init__(self, input_size, action_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
