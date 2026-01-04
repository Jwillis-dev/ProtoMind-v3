class Tool:
    """
    Abstract tool representation.
    """

    def __init__(self, name, affordances):
        self.name = name
        self.affordances = affordances  # what it enables
        self.history = []

    def __repr__(self):
        return f"<Tool {self.name} {self.affordances}>"
    
    def record_use(self, context=None):
        self.history.append(context)

