import abc

class ExplorationStrategy(object):

    def __init__(self, env_spec):
        self._env_spec = env_spec

    @abc.abstractclassmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def add_exploration(self, t, actions):
        """
        :param t: step in training (for schedule)
        :param actions: [...., action_dim]
        :return: actions + noise
        """
        raise NotImplementedError
