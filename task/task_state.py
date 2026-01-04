class TaskState:
    """
    Holds evolving task context.
    No rewards. No scores. No success flags.
    """

    def __init__(self, raw_task):
        self.raw_task = raw_task

        self.abstract_goal = None
        self.constraints = []
        self.hypotheses = []

        self.tools = []
        self.simulations = []

        self.history = []  # traceability only
