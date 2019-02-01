import abc

class Labeller:

    def __init__(self, env_spec, policy):
        self._env_spec = env_spec
        self._policy = policy

    @abc.abstractmethod
    def label(self, observations, goals):
        """
        :return goals
        """
        raise NotImplementedError