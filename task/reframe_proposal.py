# protomind/task/reframe_proposal.py

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


ReframeLevel = Literal[1, 2, 3]


"""
Data-only reframe proposal for ProtoMind.
"""

@dataclass
class ReframeProposal:
    """
    A REFRAME IS DATA, NOT ACTION.

    This object describes a proposed transformation of a task
    after failure or stagnation. It does NOT execute anything.
    It can be accepted, rejected, logged, or escalated.

    HARD RULES:
    - Reframing must never claim success
    - Reframing must never reduce external difficulty
    - Reframing must always preserve truth of failure
    """

    # --- Required ---
    level: ReframeLevel
    reason: str
    failed_component: str

    # --- What changes ---
    representation_change: Optional[str] = None
    capability_needed: Optional[str] = None
    assumption_shift: Optional[str] = None

    # --- What must NOT change ---
    preserves_goal: bool = True
    preserves_environment_truth: bool = True

    # --- Diagnostics / metadata ---
    evidence: Optional[Dict[str, Any]] = None

    def is_valid(self) -> bool:
        """
        Validate the reframe proposal.
        Invalid reframes must be rejected by runtime.
        """
        # Level must be known
        if self.level not in (1, 2, 3):
            return False

        # Must have a reason and failed component
        if not self.reason or not self.failed_component:
            return False

        # Level-specific requirements
        if self.level == 1:
            return self.representation_change is not None

        if self.level == 2:
            return self.capability_needed is not None

        if self.level == 3:
            # Level 3 is dangerous: must preserve truth
            return self.preserves_environment_truth

        return False
