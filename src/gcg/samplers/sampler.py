import numpy as np

from gcg.envs.env import Env

class Sampler(object):
    def __init__(self, env, policy, replay_pool):
        self._env = OneStepDelayAndHorizonEnvWrapper(env)
        self._policy = policy
        self._replay_pool = replay_pool

        self._curr_observation, self._curr_goal = None, None

    @property
    def env(self):
        return self._env.env

    def step(self, step, take_random_actions=False, explore=True, action=None, goal_override=None, use_labeller=True, **kwargs):
        if self._curr_observation is None or self._curr_goal is None:
            self.reset()

        """ Takes one step and adds to replay pool """
        assert (self._env is not None)
        obs_im, obs_vec = curr_obs = self._curr_observation
        curr_goal = self._curr_goal
        if goal_override is not None:
            curr_goal = goal_override

        ### store last observations and get encoded
        curr_goal = self._replay_pool.store_observation(step, (obs_im, obs_vec), curr_goal, use_labeller=use_labeller)
        encoded_observation = self._replay_pool.encode_recent_observation()

        ### get actions
        action_info = dict()
        if take_random_actions:
            assert (action is None)
            action = self._env.action_selection_space.sample()
        else:
            if action is None:
                action, _, action_info = self._policy.get_action(
                    step=step,
                    current_episode_step=self._env.current_episode_step,
                    observation=encoded_observation,
                    goal=curr_goal,
                    explore=explore)

        ### take step
        next_observation, goal, reward, done, env_info = self._env.step(action, **kwargs)
        env_info.update(action_info)

        if done:
            self._policy.reset_get_action()

        ### add to replay pool
        self._replay_pool.store_effect(action, reward, done, env_info)

        self._curr_observation = next_observation
        self._curr_goal = goal

        return curr_obs, curr_goal, action, reward, done, env_info

    def reset(self, **kwargs):
        assert (self._env is not None)

        self._curr_observation, self._curr_goal = self._env.reset(**kwargs)
        self._replay_pool.force_done()

    def get_current_goal(self, labeller=None):
        if labeller:
            return labeller.label(([self._curr_observation[0]], [self._curr_observation[1]]), [self._curr_goal])[0]
        else:
            return self._curr_goal

    @property
    def is_done(self):
        return self._env.is_done


class OneStepDelayAndHorizonEnvWrapper(Env):

    def __init__(self, env):
        self._env = env

        self._t = 0
        self._skip = False
        assert(np.isfinite(self._env.horizon))

    @property
    def env(self):
        return self._env

    def step(self, action, **kwargs):
        if self._skip:
            next_observation, goal = self._env.reset(**kwargs)
            reward = 0
            done = True
            env_info = dict()
            self._t = 0
        else:
            next_observation, goal, reward, done, env_info = self._env.step(action, **kwargs)
            self._t += 1

        if self._t >= self._env.horizon:
            done = True

        if self._skip:
            self._skip = False
        elif done:
            # delay done by one timestep
            self._skip = True
            done = False

        return next_observation, goal, reward, done, env_info

    def reset(self, **kwargs):
        obs, goal = self._env.reset(**kwargs)
        self._t = 0
        self._skip = False
        return obs, goal

    @property
    def action_selection_space(self):
        return self._env.action_selection_space

    @property
    def current_episode_step(self):
        return self._t

    @property
    def is_done(self):
        return self._skip

