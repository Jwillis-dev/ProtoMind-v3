# protomind/task/executive.py

from protomind.task.tool import Tool
from protomind.task.simulation import Simulation
from protomind.task.capability_gap import CapabilityGapDetector
from protomind.task.tool_evolver import ToolEvolver

class Executive:
    """
    Handles task execution, manages tools, detects capability gaps,
    and evolves tools when necessary.
    """

    def __init__(self, environment, tools):
        self.environment = environment
        self.tools = tools
        self.simulator = Simulation(environment)
        self.gap_detector = CapabilityGapDetector(stall_threshold=3)
        self.tool_evolver = ToolEvolver()

        # Internal state for step management
        self._step_count = 0
        self._max_steps = 10  # can be adjusted per task

    def step(self):
        """
        Perform one step of the task.
        Returns True when task is complete.
        """
        self._step_count += 1

        # Pick a tool for this step (naive round-robin for now)
        tool_idx = self._step_count % len(self.tools)
        tool = self.tools[tool_idx]

        # Simulate tool in environment
        observation = self.simulator.run(tool)

        # Check for capability gaps
        if self.gap_detector.update(observation):
            print(f"[Executive] Capability gap detected at step {self._step_count}")
            self.evolve_tools(tool)

        # Simple task completion condition
        if self._step_count >= self._max_steps:
            print("[Executive] Task completed")
            return True

        return False

    def evolve_tools(self, tool):
        """
        Use ToolEvolver to generate new candidates and update toolset.
        """
        candidates = self.tool_evolver.generate(tool)
        print(f"[Executive] Evolved {len(candidates)} new candidate tools")

        # Test candidates in simulation
        best_tool = tool
        best_score = -float("inf")
        for candidate in candidates:
            sim_trace = self.simulator.run(candidate)
            # Score = improvement in environment progress (simple metric)
            score = self.evaluate_trace(sim_trace)
            if score > best_score:
                best_score = score
                best_tool = candidate

        # Replace or add the best candidate to toolset
        if best_tool != tool:
            print(f"[Executive] Replacing tool '{tool.name}' with evolved '{best_tool.name}'")
            self.tools[self.tools.index(tool)] = best_tool

    def evaluate_trace(self, trace):
        """
        Evaluate simulation trace for effectiveness.
        This is minimal, just checks progress of the 'pos' field if present.
        """
        try:
            start = trace[0][1].get("pos", 0)
            end = trace[-1][1].get("pos", 0)
            return end - start
        except Exception:
            return 0
