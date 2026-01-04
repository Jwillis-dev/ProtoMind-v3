# protomind/core/runtime.py

from protomind.task.executive import Executive
from protomind.task.simulation import Simulation
from protomind.task.tool import Tool
from protomind.environment.environment import Environment

class ProtoMindRuntime:
    def __init__(self, environment: Environment, abstraction_size: int, hidden_size: int):
        # Executive picks candidate tools/actions
        self.executive = Executive(abstraction_size, hidden_size)
        # Environment & Simulation
        self.environment = environment
        self.simulation = Simulation(environment)
        # Active task/tool placeholder
        self.active_tool: Tool | None = None

    def tick(self, candidates: list[Tool]):
        """
        Single runtime step.
        If idle, pick a candidate tool and run simulation.
        """
        if self.active_tool is None:
            self.run_idle(candidates)
        else:
            self.run_active()

    def run_idle(self, candidates: list[Tool]):
        """
        Pick a tool from candidates and simulate a step.
        """
        # Pick tool via Executive
        self.active_tool = self.executive.step(candidates)
        # Run tool in simulation
        observation = self.simulation.run(self.active_tool)
        # Check for capability gap
        gap_detected = self.executive.check_gap(observation)
        if gap_detected:
            print(f"[ProtoMindRuntime] Capability gap detected with tool {self.active_tool.name}")
            # Reset executive or take corrective measures
            self.executive.reset()
        # After simulation, tool is idle again
        self.active_tool = None

    def run_active(self):
        """
        For future tasks that take multiple steps (currently one-shot).
        """
        pass
