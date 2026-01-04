# protomind/task/affordance_probe.py

from typing import Any, Iterable


class AffordanceProbe:
    """
    Tests whether the environment can change from the current state.
    This is NOT a planner, scorer, or learner.
    """

    def __init__(self, test_actions: Iterable[Any]):
        """
        test_actions: a small set of actions to probe the environment
        """
        self.test_actions = list(test_actions)

    def is_change_possible(self, environment, current_observation) -> bool:
        """
        Returns True if ANY test action changes the observation.
        """
        for action in self.test_actions:
            obs = environment.step(action)
            if obs != current_observation:
                return True
        return False
