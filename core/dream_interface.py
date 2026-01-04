# protomind/core/dream_interface.py

from pathlib import Path
import torch

# HARD CONTRACT:
# DreamState overwrites deltas.
# ProtoMind reads them only.
# No negotiation. No history, no accumulation.

DREAM_DELTA_PATH = Path("/home/jwillis/mnt/nvme/Delta/delta_latest.pt")


class _ImmutableDelta:
    """
    Proxy object that forbids mutation or saving.
    Only allows read access to the latest delta.
    """
    def __init__(self, path: Path):
        self._path = path

    def load(self):
        """Load the latest subconscious delta. Silent fail if missing or corrupted."""
        if not self._path.exists():
            return None
        try:
            return torch.load(self._path, map_location="cpu")
        except Exception:
            return None

    def __setattr__(self, key, value):
        if key in ("_path",):
            super().__setattr__(key, value)
        else:
            raise AttributeError("Dream delta is read-only. Cannot assign attributes.")

    def __delattr__(self, item):
        raise AttributeError("Dream delta is read-only. Cannot delete attributes.")

    def save(self, *args, **kwargs):
        raise PermissionError("Dream delta is read-only. Saving is forbidden.")


# Single immutable instance to import everywhere
dream_delta = _ImmutableDelta(DREAM_DELTA_PATH)


def load_dream_delta():
    """Convenience wrapper to load the latest delta."""
    return dream_delta.load()
