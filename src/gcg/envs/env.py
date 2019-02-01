import abc

from gcg.envs.env_spec import EnvSpec

class Env:
    """ What subclasses must implement """

    @abc.abstractmethod
    def __init__(self, params={}):
        self.observation_im_space = None
        self.action_space = None
        self.action_selection_space = None
        self.observation_vec_spec = None
        self.action_spec = None
        self.action_selection_spec = None
        self.goal_spec = None
        self.spec = EnvSpec(
            observation_im_space=self.observation_im_space,
            action_space=self.action_space,
            action_selection_space=self.action_selection_space,
            observation_vec_spec=self.observation_vec_spec,
            action_spec=self.action_spec,
            action_selection_spec=self.action_selection_spec,
            goal_spec=self.goal_spec
        )

    @abc.abstractmethod
    def step(self, action):
        next_observation = None
        goal = None
        reward = None
        done = None
        env_info = None
        return next_observation, goal, reward, done, env_info

    @abc.abstractmethod
    def reset(self):
        observation = None
        goal = None
        return observation, goal

    # @abc.abstractproperty
    # def horizon(self):
    #     raise NotImplementedError

    def log(self, prefix=''):
        pass
