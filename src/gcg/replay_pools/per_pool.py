import abc
import numpy as np

from gcg.replay_pools.replay_pool import ReplayPool

class PERPool(ReplayPool):
    @abc.abstractmethod
    def __init__(self, env_spec, obs_history_len, N, size,
                 labeller=None, save_rollouts=False, save_rollouts_observations=True, save_env_infos=False):
        super(PERPool, self).__init__(env_spec=env_spec,
                                      obs_history_len=obs_history_len,
                                      N=N,
                                      size=size,
                                      labeller=labeller,
                                      save_rollouts=save_rollouts,
                                      save_rollouts_observations=save_rollouts_observations,
                                      save_env_infos=save_env_infos)
        self._priorities = np.zeros((self._size), dtype=np.float32)
        self._init_ps()

    ###############################
    ### Priority Data Structure ###
    ###############################

    @abc.abstractmethod
    def _init_ps(self):
        pass

    @abc.abstractmethod
    def update_priorities(self, indices, priorities):
        pass

    ########################
    ### Sample from pool ###
    ########################

    @abc.abstractmethod
    def _sample_start_indices(self, batch_size):
        return np.ones(batch_size) * np.nan, np.ones(batch_size) * np.nan

    ###################
    ### Add to pool ###
    ###################

    def store_effect(self, action, reward, done, env_info, flatten_action=True, update_log_stats=True):
        self.update_priorities(np.array([self._index]), np.array([np.inf if not done else 0.0]))
        ReplayPool.store_effect(self, action, reward, done, env_info, flatten_action=flatten_action, update_log_stats=update_log_stats)

    ########################
    ### Remove from pool ###
    ########################

    def trash_current_rollout(self):
        new_indices = self._new_indices
        self._priorities[new_indices] = 0.
        self.update_priorities(new_indices, np.zeros(len(self._new_indices)))
        ReplayPool.trash_current_rollout(self)
