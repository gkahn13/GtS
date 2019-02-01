import time
from collections import defaultdict

import numpy as np

from gcg.misc import utils
from gcg.data.logger import logger
from gcg.data import mypickle

class ReplayPool(object):

    def __init__(self, env_spec, obs_history_len, N, size,
                 labeller=None, save_rollouts=False, save_rollouts_observations=True, save_env_infos=False):
        """
        :param env_spec: for observation/action dimensions
        :param N: horizon length
        :param size: size of pool
        :param obs_history_len: how many previous obs to include when sampling? (= 1 is only current observation)
        :param save_rollouts: for debugging
        """
        self._env_spec = env_spec
        self._N = N
        self._size = int(size)
        self._obs_history_len = obs_history_len
        self._labeller = labeller
        self._save_rollouts = save_rollouts
        self._save_rollouts_observations = save_rollouts_observations
        self._save_env_infos = save_env_infos

        ### buffer
        self._obs_im_shape = self._env_spec.observation_im_space.shape
        assert(len(self._obs_im_shape) == 3)
        self._obs_im_dim = np.prod(self._obs_im_shape)
        self._obs_vec_dim = len(list(self._env_spec.observation_vec_spec.keys()))
        self._goal_dim = len(list(self._env_spec.goal_spec.keys()))
        self._action_dim = self._env_spec.action_space.flat_dim
        self._steps = np.empty((self._size,), dtype=np.int32)
        self._observations_im = np.empty((self._size, self._obs_im_dim), dtype=np.uint8)
        self._observations_vec = np.ones((self._size, self._obs_vec_dim), dtype=np.float32)
        self._goals = np.nan * np.ones((self._size, self._goal_dim), dtype=np.float32)
        self._actions = np.nan * np.ones((self._size, self._action_dim), dtype=np.float32)
        self._rewards = np.nan * np.ones((self._size,), dtype=np.float32)
        self._dones = np.ones((self._size,), dtype=bool) # initialize as all done
        self._env_infos = np.empty((self._size,), dtype=np.object)
        self._index = 0
        self._curr_size = 0
        self._data_step = 0

        ### logging
        self._last_done_index = 0
        self._log_stats = defaultdict(list)
        self._log_paths = []
        self._last_get_log_stats_time = None

    def __len__(self):
        return self._curr_size

    @property
    def size(self):
        return self._size

    def _get_indices(self, start, end):
        start = start % self._size
        end = end % self._size
        if start <= end:
            return np.arange(start, end)
        elif start > end:
            return np.arange(start - self._size, end)

    @property
    def _new_indices(self):
        return self._get_indices(self._last_done_index, self._index)

    @property
    def _prev_index(self):
        return (self._index - 1) % self._size

    ###################
    ### Add to pool ###
    ###################

    def store_observation(self, step, observation, goal, use_labeller=True):
        self._data_step = step
        self._steps[self._index] = step
        obs_im_full, obs_vec = observation

        if self._labeller and use_labeller:
            assert (np.any(np.isnan(goal)))
            goal = self._labeller.label(([obs_im_full], [obs_vec]), [goal])[0]

        #import IPython; IPython.embed()
        obs_im = utils.imresize(obs_im_full, self._obs_im_shape)
        assert(obs_im.shape == self._obs_im_shape)
        self._observations_im[self._index, :] = obs_im.reshape((self._obs_im_dim,))
        self._observations_vec[self._index, :] = obs_vec.reshape((self._obs_vec_dim,))
        self._goals[self._index, :] = goal

        return goal

    def _encode_observation(self, index):
        """ Encodes observation starting at index by concatenating obs_history_len previous """
        indices = self._get_indices(index - self._obs_history_len + 1, (index + 1) % self._size)  # plus 1 b/c inclusive
        observations_im = self._observations_im[indices]
        observations_vec = self._observations_vec[indices]
        dones = self._dones[indices]

        encountered_done = False
        for i in range(len(dones) - 2, -1, -1):  # skip the most current frame since don't know if it's done
            encountered_done = encountered_done or dones[i]
            if encountered_done:
                observations_im[i, ...] = 0.
                observations_vec[i, ...] = 0.

        return observations_im, observations_vec

    def encode_recent_observation(self):
        return self._encode_observation(self._index)

    def _done_update(self, update_log_stats=True):
        if self._last_done_index == self._index:
            return

        ### update log stats
        if update_log_stats:
            self._update_log_stats()

        self._last_done_index = self._index

    def store_effect(self, action, reward, done, env_info, flatten_action=True, update_log_stats=True):
        self._actions[self._index, :] = self._env_spec.action_space.flatten(action) if flatten_action else action
        self._rewards[self._index] = reward

        self._dones[self._index] = done
        self._env_infos[self._index] = env_info if self._save_env_infos else None
        self._curr_size = max(self._curr_size, self._index + 1)
        self._index = (self._index + 1) % self._size

        ### compute values
        if done:
            self._done_update(update_log_stats=update_log_stats)

    def force_done(self):
        if len(self) == 0:
            return

        self._dones[self._prev_index] = True
        self._done_update()

    def _store_rollout(self, start_step, rollout):
        if not rollout['dones'][-1]:
            logger.warn('Rollout not ending in done. Not being added to replay pool.')
            return

        r_len = len(rollout['dones'])
        for i in range(r_len):
            obs_im = np.reshape(rollout['observations_im'][i], self._obs_im_shape)
            obs_vec = rollout['observations_vec'][i]
            if 'goals' in rollout:
                goal = rollout['goals'][i]
            else:
                goal = np.zeros((self._goal_dim,))

            self.store_observation(start_step + i, (obs_im, obs_vec), goal)

            self.store_effect(rollout['actions'][i],
                              rollout['rewards'][i],
                              rollout['dones'][i],
                              None, # TODO rollout['env_infos'][i],
                              update_log_stats=False)

    def store_rollouts(self, rlist, max_to_add=None):
        """
        rlist can be a list of pkl filenames, or rollout dictionaries
        """
        step = len(self)

        for rlist_entry in rlist:
            if type(rlist_entry) is str:
                rollouts = mypickle.load(rlist_entry)['rollouts']
            elif issubclass(type(rlist_entry), dict):
                rollouts = [rlist_entry]
            else:
                raise NotImplementedError

            for rollout in rollouts:
                r_len = len(rollout['dones'])
                if max_to_add is not None and step + r_len >= max_to_add:
                    return

                self._store_rollout(step, rollout)
                step += r_len

    def update_priorities(self, indices, priorities):
        pass

    ########################
    ### Remove from pool ###
    ########################

    def trash_current_rollout(self):
        new_indices = self._new_indices
        self._actions[new_indices, :] = np.nan
        self._rewards[new_indices] = np.nan
        self._dones[new_indices] = True
        self._env_infos[new_indices] = np.object
        self._index = self._last_done_index

        return len(new_indices)

    ########################
    ### Sample from pool ###
    ########################

    def can_sample(self, batch_size=1):
        return len(self) > self._obs_history_len and len(self) > self._N

    def _sample_start_indices(self, batch_size):
        valid_indices = np.logical_not(self._dones[:self._curr_size])
        if len(self._new_indices) > 0:
            valid_indices[self._new_indices[-1]] = False

        start_indices = np.random.choice(np.where(valid_indices)[0], batch_size)
        weights = np.ones(len(start_indices), dtype=np.float32)
        
        return start_indices, weights

    def sample(self, batch_size, include_env_infos=False):
        """
        :return observations, actions, and rewards of horizon H+1
        """
        if not self.can_sample():
            return None

        steps, observations_im, observations_vec, goals, actions, rewards, dones, env_infos = [], [], [], [], [], [], [], []
        
        start_indices, weights = self._sample_start_indices(batch_size)

        for start_index in start_indices:
            indices = self._get_indices(start_index, (start_index + self._N + 1) % self._size)
            steps_i = self._steps[indices]
            obs_im_i, obs_vec_i = self._encode_observation(start_index)
            observations_im_i = np.vstack([obs_im_i, self._observations_im[indices[1:]]])
            observations_vec_i = np.vstack([obs_vec_i, self._observations_vec[indices[1:]]])
            goals_i = self._goals[indices]
            actions_i = self._actions[indices]
            rewards_i = self._rewards[indices]
            dones_i = self._dones[indices]
            env_infos_i = self._env_infos[indices] if include_env_infos else [None] * len(dones_i)
            if dones_i[0]:
                raise Exception('Should not ever happen')
            if np.any(dones_i):
                # H = 3
                # observations = [0 1 2 3]
                # actions = [10 11 12 13]
                # rewards = [20 21 22 23]
                # dones = [False True False False]

                d_idx = np.argmax(dones_i)
                rewards_i[d_idx:len(dones_i)] = 0.
                dones_i[d_idx:len(dones_i)] = True
                actions_i[d_idx:len(dones_i)] = self._env_spec.action_space.flatten_n(self._env_spec.action_space.sample_n(len(dones_i) - d_idx))
                goals_i[d_idx:len(dones_i)] = goals_i[d_idx-1]

                # observations = [0 1 2 3]
                # actions = [10 11 rand rand]
                # rewards = [20 21 0 0]
                # dones = [False True True True]

            steps.append(np.expand_dims(steps_i, 0))
            observations_im.append(np.expand_dims(observations_im_i, 0))
            observations_vec.append(np.expand_dims(observations_vec_i, 0))
            goals.append(np.expand_dims(goals_i, 0))
            actions.append(np.expand_dims(actions_i, 0))
            rewards.append(np.expand_dims(rewards_i, 0))
            dones.append(np.expand_dims(dones_i, 0))
            env_infos.append(np.expand_dims(env_infos_i, 0))

        steps = np.vstack(steps)
        observations_im = np.vstack(observations_im)
        observations_vec = np.vstack(observations_vec)
        goals = np.vstack(goals)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)
        env_infos = np.vstack(env_infos)

        return start_indices, weights, steps, (observations_im, observations_vec), goals, actions, rewards, dones, env_infos

    def sample_all_generator(self, batch_size, include_env_infos=False):
        if not self.can_sample():
            return

        start_indices, weights, steps, observations_im, observations_vec, goals, actions, rewards, dones, env_infos = \
            [], [], [], [], [], [], [], [], [], []

        for start_index in range(len(self) - 1):
            indices = self._get_indices(start_index, (start_index + self._N + 1) % self._curr_size)

            steps_i = self._steps[indices]
            obs_im_i, obs_vec_i = self._encode_observation(start_index)
            observations_im_i = np.vstack([obs_im_i, self._observations_im[indices[1:]]])
            observations_vec_i = np.vstack([obs_vec_i, self._observations_vec[indices[1:]]])
            goals_i = self._goals[indices]
            actions_i = self._actions[indices]
            rewards_i = self._rewards[indices]
            dones_i = self._dones[indices]
            env_infos_i = self._env_infos[indices] if include_env_infos else [None] * len(dones_i)

            if dones_i[0]:
                continue

            start_indices.append(start_index)
            weights.append(1.)
            steps.append(np.expand_dims(steps_i, 0))
            observations_im.append(np.expand_dims(observations_im_i, 0))
            observations_vec.append(np.expand_dims(observations_vec_i, 0))
            goals.append(np.expand_dims(goals_i, 0))
            actions.append(np.expand_dims(actions_i, 0))
            rewards.append(np.expand_dims(rewards_i, 0))
            dones.append(np.expand_dims(dones_i, 0))
            env_infos.append(np.expand_dims(env_infos_i, 0))

            if len(steps) >= batch_size:
                yield np.asarray(start_indices), np.asarray(weights),\
                      np.vstack(steps), (np.vstack(observations_im), np.vstack(observations_vec)), np.vstack(goals), \
                      np.vstack(actions), np.vstack(rewards), np.vstack(dones), np.vstack(env_infos)
                start_indices, weights, steps, observations_im, observations_vec, goals, actions, rewards, dones, env_infos = \
                    [], [], [], [], [], [], [], [], [], []

        if len(start_indices) > 0:
            yield np.asarray(start_indices), np.asarray(weights), \
                  np.vstack(steps), (np.vstack(observations_im), np.vstack(observations_vec)), np.vstack(goals), \
                  np.vstack(actions), np.vstack(rewards), np.vstack(dones), np.vstack(env_infos)

    def sample_rollouts(self, num_rollouts):
        assert (self._save_rollouts_observations)

        ### get start indices
        start_indices = []
        if len(self) < self._size:
            start_indices.append(0)
            start_indices += ((self._dones[:self._curr_size].nonzero()[0] + 1) % self._curr_size).tolist()

        rollouts = []
        for i in range(num_rollouts):
            start_index_index = np.random.randint(0, len(start_indices) - 1)
            start_index = start_indices[start_index_index]
            end_index = start_indices[(start_index_index + 1) % len(start_indices)]

            indices = self._get_indices(start_index, end_index)

            rollout = {
                'steps': self._steps[indices],
                'observations_im': self._observations_im[indices],
                'observations_vec': self._observations_vec[indices],
                'goals': self._goals[indices],
                'actions': self._actions[indices],
                'rewards': self._rewards[indices],
                'dones': self._dones[indices],
                'env_infos': None,
            }

            assert (rollout['dones'][-1] == True)
            assert (rollout['dones'].sum() == 1)
            rollouts.append(rollout)

        return rollouts

    ###############
    ### Logging ###
    ###############

    def _update_log_stats(self):
        indices = self._new_indices

        ### update log
        rewards = self._rewards[indices]
        # Use 2nd to last because last reward should be dummy done
        self._log_stats['FinalReward'].append(rewards[-2])
        self._log_stats['AvgReward'].append(np.mean(rewards))
        self._log_stats['CumReward'].append(np.sum(rewards))
        self._log_stats['EpisodeLength'].append(len(rewards))

        ## update paths
        if self._save_rollouts:

            if self._save_env_infos:
                env_infos = self._env_infos[indices][:-1] # b/c last will be empty
                env_info_keys = env_infos[0].keys()
                env_infos = {k: np.array([ei[k] for ei in env_infos]) for k in env_info_keys}
            else:
                env_infos = None

            self._log_paths.append({
                'steps': self._steps[indices],
                'observations_im': self._observations_im[indices] if self._save_rollouts_observations else None,
                'observations_vec': self._observations_vec[indices] if self._save_rollouts_observations else None,
                'goals': self._goals[indices],
                'actions': self._actions[indices],
                'rewards': self._rewards[indices],
                'dones': self._dones[indices],
                'env_infos': env_infos,
            })

        # clear env_infos so we save memory, but won't be able to access with sample(...)
        self._env_infos[indices] = np.object

    def log(self, prefix=''):
        self._log_stats['Time'] = [time.time() - self._last_get_log_stats_time] if self._last_get_log_stats_time else [0.]

        logger.record_tabular(prefix+'CumRewardMean', np.mean(self._log_stats['CumReward']))
        logger.record_tabular(prefix+'CumRewardStd', np.std(self._log_stats['CumReward']))
        logger.record_tabular(prefix+'AvgRewardMean', np.mean(self._log_stats['AvgReward']))
        logger.record_tabular(prefix+'AvgRewardStd', np.std(self._log_stats['AvgReward']))
        logger.record_tabular(prefix+'FinalRewardMean', np.mean(self._log_stats['FinalReward']))
        logger.record_tabular(prefix+'FinalRewardStd', np.std(self._log_stats['FinalReward']))
        logger.record_tabular(prefix+'EpisodeLengthMean', np.mean(self._log_stats['EpisodeLength']))
        logger.record_tabular(prefix+'EpisodeLengthStd', np.std(self._log_stats['EpisodeLength']))

        logger.record_tabular(prefix+'NumEpisodes', len(self._log_stats['EpisodeLength']))
        logger.record_tabular(prefix+'Time', np.mean(self._log_stats['Time']))

        self._last_get_log_stats_time = time.time()
        self._log_stats = defaultdict(list)

    def get_recent_rollouts(self):
        paths = self._log_paths
        self._log_paths = []
        return paths

    @property
    def finished_storing_rollout(self):
        return (self._last_done_index == self._index)