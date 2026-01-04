# protomind/task/escalation.py

from enum import Enum, auto


class EscalationDecision(Enum):
    REFRAME_TASK = auto()
    REQUEST_NEW_TOOL_CLASS = auto()
    ABANDON_TASK = auto()


class TaskEscalator:
    """
    Handles what ProtoMind does when a task is locally impossible.
    This is NOT a reward system.
    This is epistemic control flow.
    """

    def decide(self, context: dict) -> EscalationDecision:
        """
        Decide how to proceed after impossibility is detected.

        context may include:
        - task
        - environment
        - recent observations
        - tools tried
        """

        # v1 logic: conservative and honest

        if context.get("environment_immutable", False):
            return EscalationDecision.REFRAME_TASK

        if context.get("tools_exhausted", False):
            return EscalationDecision.REQUEST_NEW_TOOL_CLASS

        return EscalationDecision.ABANDON_TASK
