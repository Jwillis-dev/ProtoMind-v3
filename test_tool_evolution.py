# test_tool_evolution.py

from __future__ import annotations

from task.tool import Tool
from task.executive import Executive
from environment.environment import Environment

class DummyEnv(Environment):
    def __init__(self):
        self.pos = 0
        self.name = "dummy_env"

    def reset(self):
        self.pos = 0
        return {"pos": self.pos}

    def step(self, action):
        aff = (action or {}).get("affordances", {}) or {}
        speed = float(aff.get("speed", 0.0))
        self.pos += speed
        if self.pos > 5:
            self.pos = 5
        return {"pos": self.pos}


def test_run():
    env = DummyEnv()
    tools = [Tool("move", {"speed": 1.0})]
    ex = Executive(env, tools)

    # Run until abandon or step cap
    for _ in range(50):
        ended = ex.step()
        if ended:
            break

    # Should either abandon or keep running; this is a smoke test.
    assert ex.tools, "Tools list should not be empty."

if __name__ == "__main__":
    test_run()
    print("OK")
