# protomind/main.py

from protomind.core.runtime import ProtoMindRuntime
from protomind.task.tool import Tool
from protomind.task.executive import Executive
from protomind.environment.environment import DummyEnv  # Or GridEnvironment

def build_tools():
    """
    Define starting toolset for ProtoMind.
    """
    return [
        Tool("move", {"speed": 1.0, "can_turn": True}),
        Tool("move_var", {"speed": 1.0, "can_turn": True}),
        # Add more tools here
    ]

def build_task(environment, tools):
    """
    Build an Executive task using the environment and toolset.
    """
    return Executive(environment, tools)

def main():
    # --- Build environment ---
    env = DummyEnv()  # Replace with GridEnvironment(width, height) if needed

    # --- Build tools ---
    tools = build_tools()

    # --- Build executive task ---
    exec_task = build_task(env, tools)

    # --- Build runtime ---
    # For now, dummy placeholders for dream_engine and evolutionary_loop
    class DummyDreamEngine:
        def dream_cycle(self):
            print("[DreamEngine] idle dream cycle")

    dummy_dream = DummyDreamEngine()
    runtime = ProtoMindRuntime(dream_engine=dummy_dream, evolutionary_loop=None)

    # --- Schedule executive task ---
    runtime.schedule_task(exec_task)

    # --- Run runtime loop ---
    for tick_num in range(15):  # Run a fixed number of ticks for demo
        print(f"\n[Runtime] Tick {tick_num}")
        runtime.tick()

if __name__ == "__main__":
    main()
