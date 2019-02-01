from gcg.envs.env import Env
from gcg.envs.env_spec import EnvSpec
from collections import OrderedDict
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
import numpy as np


class TfrecordsEnv(Env):
    """ What subclasses must implement """

    def __init__(self, params={}):
        self._yaw_limits = params['yaw_limits']
        self._obs_shape = params['obs_shape']
        self._horizon = params['horizon']

        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()

        self.action_spec['yaw'] = Box(low=-180, high=180)

        self.action_space = Box(low=np.array([self.action_spec['yaw'].low[0]]),
                                high=np.array([self.action_spec['yaw'].high[0]]))

        self.action_selection_spec['yaw'] = Box(low=self._yaw_limits[0], high=self._yaw_limits[1])

        self.action_selection_space = Box(low = np.array([self.action_selection_spec['yaw'].low[0]]), high = np.array([self.action_selection_spec['yaw'].high[0]]))

        self.observation_im_space = Box(low=0, high=255, shape=self._obs_shape)
        self.observation_vec_spec['coll'] = Discrete(1)
        self.spec = EnvSpec(
            observation_im_space=self.observation_im_space,
            action_space=self.action_space,
            action_selection_space=self.action_selection_space,
            observation_vec_spec=self.observation_vec_spec,
            action_spec=self.action_spec,
            action_selection_spec=self.action_selection_spec,
            goal_spec=self.goal_spec
        )

    @property
    def horizon(self):
        return self._horizon

    def step(self, action):
        next_observation = None
        goal = None
        reward = None
        done = None
        env_info = None
        return next_observation, goal, reward, done, env_info

    def reset(self):
        observation = None
        goal = None
        return observation, goal

    def log(self, prefix=''):
        pass