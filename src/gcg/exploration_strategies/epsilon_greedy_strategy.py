import numpy as np

from gcg.exploration_strategies.exploration_strategy import ExplorationStrategy
from gcg.misc import schedules

class EpsilonGreedyStrategy(ExplorationStrategy):
    """
    Takes random action with probability epsilon
    """
    def __init__(self, env_spec, endpoints, outside_value):
        super(EpsilonGreedyStrategy, self).__init__(env_spec)

        self.schedule = schedules.PiecewiseSchedule(endpoints=endpoints, outside_value=outside_value)

    def reset(self):
        pass

    def add_exploration(self, t, actions):
        aspace = self._env_spec.action_space
        actions = np.asarray(actions)
        assert (len(actions.shape) > 1)
        assert (actions.shape[-1] == aspace.flat_dim)
        other_shape = list(actions.shape[:-1])

        aspace_select = self._env_spec.action_selection_space
        actions = np.reshape(actions, (-1, aspace.flat_dim))

        for i in range(len(actions)):
            if np.random.random() < self.schedule.value(t):
                actions[i] = aspace_select.sample()

        actions = np.reshape(actions, other_shape + [aspace.flat_dim])
        return actions
