
# main.py

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


def main():
    env = DummyEnv()
    tools = [Tool("move", {"speed": 1.0})]
    ex = Executive(env, tools)

    for step in range(50):
        ended = ex.step()
        print(f"[Step {step}] tools={ex.tools} abandoned={ex.abandoned}")
        if ended:
            print("Task episode ended (abandoned).")
            break


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
