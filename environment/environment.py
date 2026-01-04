class Environment:
    """
    Minimal environment interface.
    """

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """
        Apply an action.
        Returns observation (opaque).
        """
        raise NotImplementedError
