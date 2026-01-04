# protomind/core/snapshot_store.py
import torch
from typing import Dict
from .versioning import BrainVersion

class SnapshotStore:
    """
    Stores brain weights + version metadata.
    Brains never access this directly.
    """
    def __init__(self):
        self.snapshots: Dict[str, Dict] = {}

    def save(self, brain, version: BrainVersion):
        self.snapshots[version.id] = {
            "state_dict": {k: v.detach().cpu() for k, v in brain.state_dict().items()},
            "version": version
        }

    def load(self, brain, version_id: str):
        if version_id not in self.snapshots:
            raise KeyError(f"Snapshot {version_id} not found")
        brain.load_state_dict(self.snapshots[version_id]["state_dict"])

    def get_version(self, version_id: str) -> BrainVersion:
        return self.snapshots[version_id]["version"]

    def list_versions(self):
        return [v["version"] for v in self.snapshots.values()]
