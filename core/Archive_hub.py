# protomind/core/hub.py
import torch
from typing import Dict, Optional

class ProtoMindHub:
    """
    Central manager for:
      - Brain instances
      - Environment interface
      - Global state and snapshot management
    """
    def __init__(self, brain, env: 'Environment', device: Optional[str] = None):
        self.brain = brain.to(device or 'cpu')
        self.env = env
        self.device = device or 'cpu'
        self.global_state = None
        self.snapshots: Dict[str, torch.Tensor] = {}  # keyed brain snapshots
        self.step_counter = 0

    def reset(self):
        obs = self.env.reset()
        self.global_state = None
        self.step_counter = 0
        return obs

    def step(self, action):
        obs, done, info = self.env.step(action)
        self.step_counter += 1
        return obs, done, info

    def forward(self, obs):
        """
        Pass observation through the brain with current global state.
        Updates the hub's global state automatically.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, self.global_state, predicted_next_obs, *_ = self.brain.forward(
            obs_tensor, self.global_state
        )
        return logits.squeeze(0).detach(), predicted_next_obs.squeeze(0).detach()

    def save_snapshot(self, name: str):
        """Save current brain weights for later evolutionary use."""
        self.snapshots[name] = self.brain.state_dict()

    def load_snapshot(self, name: str):
        """Load a saved brain snapshot."""
        if name in self.snapshots:
            self.brain.load_state_dict(self.snapshots[name])
        else:
            raise KeyError(f"Snapshot '{name}' not found.")
