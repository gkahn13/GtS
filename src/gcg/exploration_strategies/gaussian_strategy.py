import numpy as np

from gcg.exploration_strategies.exploration_strategy import ExplorationStrategy
from gcg.envs.spaces.box import Box
from gcg.misc import schedules, utils

class GaussianStrategy(ExplorationStrategy):
    """
    Add gaussian noise
    """
    def __init__(self, env_spec, endpoints, outside_value):
        super(GaussianStrategy, self).__init__(env_spec)

        assert isinstance(env_spec.action_space, Box)
        self.schedule = schedules.PiecewiseSchedule(endpoints=endpoints, outside_value=outside_value)

    def reset(self):
        pass

    def add_exploration(self, t, actions):
        """
        :param t: step in training (for schedule)
        :param actions: [...., action_dim]
        :return: actions + noise
        """
        aspace = self._env_spec.action_space
        actions = np.asarray(actions)
        assert (len(actions.shape) > 1)
        assert (actions.shape[-1] == aspace.flat_dim)
        other_shape = list(actions.shape[:-1])

        # whiten
        mean = 0.5 * (aspace.low + aspace.high)
        scale = (aspace.high - aspace.low)
        mean = np.tile(utils.multiple_expand_dims(mean, [0] * len(other_shape)), other_shape + [1])
        scale = np.tile(utils.multiple_expand_dims(scale, [0] * len(other_shape)), other_shape + [1])
        actions = (actions - mean) / scale

        # add noise
        actions += np.random.normal(size=actions.shape) * self.schedule.value(t)

        # de-whiten
        actions = (actions * scale) + mean

        # clip
        aspace_select = self._env_spec.action_selection_space
        actions = np.clip(actions,
                          np.tile(utils.multiple_expand_dims(aspace_select.low, [0] * len(other_shape)),
                                  other_shape + [1]),
                          np.tile(utils.multiple_expand_dims(aspace_select.high, [0] * len(other_shape)),
                                  other_shape + [1]))

        return actions

