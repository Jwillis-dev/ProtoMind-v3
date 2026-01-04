# protomind/task/dummy_task.py

class DummyTask:
    """
    A minimal active task.
    Exists only to prove runtime circulation.
    """

    def __init__(self, steps=5):
        self.steps_remaining = steps

    def step(self):
        """
        Perform one unit of work.
        Returns True when task is complete.
        """
        print(f"[DummyTask] step, remaining={self.steps_remaining}")

        self.steps_remaining -= 1

        if self.steps_remaining <= 0:
            print("[DummyTask] completed")
            return True

        return False
