# protomind/test_tool_evolution.py


from protomind.task.tool import Tool
from protomind.task.tool_evolver import ToolEvolver
from protomind.task.capability_gap import CapabilityGapDetector
from protomind.task.affordance_probe import AffordanceProbe

# --- Dummy Environment ---
class DummyEnv:
    """
    Simple environment that responds to tool actions.
    """
    def __init__(self):
        self.pos = 0

    def reset(self):
        self.pos = 0
        return {"pos": self.pos}

    def step(self, action):
        """
        Apply tool's affordances.
        Accepts either a Tool object or an action dict.
        """
        if hasattr(action, 'affordances'):
            speed = action.affordances.get("speed", 0)
        elif isinstance(action, dict):
            speed = action.get("affordances", {}).get("speed", 0)
        else:
            speed = 0
        self.pos += speed
        if self.pos > 5:
            self.pos = 5
        return {"pos": self.pos}


if __name__ == "__main__":
    # Initialize environment, tool, detector, and evolver
    env = DummyEnv()
    gap_detector = CapabilityGapDetector(stall_threshold=3)
    evolver = ToolEvolver()


    # Original tool
    base_tool = Tool("move", {"speed": 1.0})

    # Define probe actions
    probe = AffordanceProbe(test_actions=[
        {"tool": base_tool.name, "affordances": base_tool.affordances},
        {"tool": "noop", "affordances": {}}
    ])

    # Reset environment
    obs = env.reset()
    print("Environment reset:", obs)

    # Run simulation loop
    for step_num in range(15):
        print(f"\nStep {step_num+1}")

        # Step environment with current tool
        obs = env.step(base_tool)
        print("Tool used:", base_tool)
        print("Observation:", obs)

        # Check for capability gap
        if gap_detector.update(obs):
            print("Capability gap detected!")

            if not probe.is_change_possible(env, obs):
                print("Environment is locally immutable. Escalating task.")
                break  # or hand control to planner / meta-system

            print("Change possible. Generating new tool variants...")
            candidates = evolver.generate(base_tool)
            base_tool = candidates[0]
            print("Tool updated to:", base_tool)
            gap_detector.reset()
