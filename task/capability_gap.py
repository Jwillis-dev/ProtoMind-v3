# protomind/task/capability_gap.py

from typing import Any, List, Tuple

class CapabilityGapDetector:
    """
    Detects when a task cannot make progress.
    This is NOT a reward or score system.
    Only flags when ProtoMind needs new capabilities or tools.
    """


    def __init__(self, stall_threshold: int = 3):
        """
        stall_threshold: how many identical steps in a row indicate a gap
        """
        self.stall_threshold = stall_threshold
        self._history: List[Any] = []

    def update(self, observation: Any) -> bool:
        """
        Record the latest observation.
        Returns True if the environment has stopped changing.
        """
        self._history.append(observation)

        if len(self._history) < self.stall_threshold:
            return False

        recent = self._history[-self.stall_threshold:]

        # World is stalled if observations are identical
        if all(obs == recent[0] for obs in recent):
            return True

        return False

    def reset(self):
        """
        Clear history (e.g., after a tool resolves the gap or task resets)
        """
        self._history.clear()
