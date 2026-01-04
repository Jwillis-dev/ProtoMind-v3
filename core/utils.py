# protomind/core/utils.py
import torch

def clone_brain(brain: torch.nn.Module) -> torch.nn.Module:
    """Deep copy a brain for mutation or parallel experiments."""
    new_brain = type(brain)(*getattr(brain, 'init_args', ()), **getattr(brain, 'init_kwargs', {}))
    new_brain.load_state_dict(brain.state_dict())
    return new_brain
