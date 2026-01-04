# protomind/task/simulation.py

from protomind.task.tool import Tool
from protomind.task.capability_gap import CapabilityGapDetector

class Simulation:
    """
    Run tools in an environment and detect capability gaps.
    """

    def __init__(self, environment):
        self.environment = environment
        self.capability_detector = CapabilityGapDetector()

    def run(self, tool: Tool, steps: int = 10):
        """
        Execute a tool in the environment for a number of steps.

        Returns a trace of (tool_state, observation) pairs.
        """
        trace = []

        # Reset environment to get initial observation
        obs = self.environment.reset()
        trace.append(("reset", obs))

        for _ in range(steps):
            # Apply tool's affordances as action
            action = {
                "tool": tool.name,
                "affordances": tool.affordances
            }

            obs = self.environment.step(action)
            trace.append((action, obs))

            # Update capability detector
            self.capability_detector.update(tool, obs)

        # Return the trace for inspection or scoring
        return trace
