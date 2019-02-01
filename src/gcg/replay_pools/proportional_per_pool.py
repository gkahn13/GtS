import numpy as np

from gcg.misc import schedules
from gcg.replay_pools.per_pool import PERPool
from gcg.data.logger import logger

class ProportionalPERPool(PERPool):

    def __init__(self, env_spec, obs_history_len, N, size,
                 labeller=None, save_rollouts=False, save_rollouts_observations=True, save_env_infos=False,
                 alpha=None, beta_schedule=None):
        assert(alpha is not None)
        assert(beta_schedule is not None)

        self._alpha = alpha
        self._beta_schedule = schedules.PiecewiseSchedule(**beta_schedule)
        self._max_pri_set = set()
        super(PERPool, self).__init__(env_spec=env_spec,
                                      obs_history_len=obs_history_len,
                                      N=N,
                                      size=size,
                                      labeller=labeller,
                                      save_rollouts=save_rollouts,
                                      save_rollouts_observations=save_rollouts_observations,
                                      save_env_infos=save_env_infos)

    ###############################
    ### Priority Data Structure ###
    ###############################

    def _init_ps(self):
        self._sum_tree = SumTree(self._size)

    def _pri_to_weights(self, priorities):
        beta = self._beta_schedule.value(self._data_step)
        weights = (1. / (self._curr_size * priorities)) ** beta
        weights = weights / np.max(weights)
        return weights

    def update_priorities(self, indices, priorities):
        self._priorities[indices] = priorities
        new_inf_indices = indices[priorities == np.inf]
        non_inf = priorities != np.inf
        non_inf_indices = indices[non_inf]
        non_inf_priorities = priorities[non_inf]

        self._max_pri_set.difference_update(non_inf_indices)
        self._max_pri_set.update(new_inf_indices)
        self._sum_tree.push_or_update(non_inf_indices, non_inf_priorities ** self._alpha)

    ########################
    ### Sample from pool ###
    ########################

    def _sample_start_indices(self, batch_size):
        assert(self.can_sample(batch_size))
        start_indices = []
        priorities = []
        for i in range(batch_size):
            ran = (np.random.random() + i) / batch_size
            index, priority = self._ran_sample_start_index(ran)
            assert(priority > 0)
            assert(not self._dones[index])
            start_indices.append(index)
            priorities.append(priority)
        weights = self._pri_to_weights(np.array(priorities))
        return np.array(start_indices), weights

    def _ran_sample_start_index(self, ran):
        assert(ran > 0)
        assert(ran <= 1)
        max_pri = self._sum_tree.max_priority if self._sum_tree.max_priority > 0 else 1.

        false_indices_set = set()
        if self._only_completed_episodes:
            false_indices_set.update(self._new_indices)
        elif len(self._new_indices) > 0:
            false_indices_set.add(self._new_indices[-1])
        false_indices_set &= self._max_pri_set
        self._max_pri_set -= false_indices_set
        max_pri_weight = len(self._max_pri_set) * max_pri
        tot_sum = self._sum_tree.total_weight + max_pri_weight
        search_val = ran * tot_sum
        if search_val <= max_pri_weight:
            value = self._max_pri_set.pop()
            pri = max_pri
        else:
            assert(self._sum_tree.can_sample)
            new_ran = float(search_val - max_pri_weight) / self._sum_tree.total_weight
            value, pri = self._sum_tree.sample(new_ran)
        self._max_pri_set |= false_indices_set
        return value, pri

class SumTree(object):
    def __init__(self, size):
        self._size = size
        self._tree = []
        self._curr_size = 0
        self._index = 0
        self._v2i = {}
        self._max_priority = 0.

    def __len__(self):
        return self._curr_size

    @property
    def total_weight(self):
        if self._curr_size == 0:
            return 0.
        else:
            return self._get_val(0)

    @property
    def max_priority(self):
        return self._max_priority

    def can_sample(self):
        return self._curr_size > 0

    def push(self, value, priority):
        assert(priority >= 0)
        assert(priority < np.inf)
        if self._curr_size  < self._size:
            self._update_max_pri(priority)
            if self._index == 0:
                self._tree.append((priority, value))
                self._v2i[value] = self._index
                self._index = 1
            else:
                up_index = (self._index - 1) // 2
                tup = self._tree[up_index]
                assert(isinstance(tup, tuple))
                self._tree.append(tup)
                self._v2i[tup[1]] = self._index
                self._tree.append((priority, value))
                self._v2i[value] = self._index + 1
                self._update_up(self._index)
                self._index += 2
            self._curr_size += 1
        else:
            logger.warn('Attempting to add data beyond Priority Sum Tree capacity!')

    def update_priority(self, value, priority):
        assert(priority >= 0)
        assert(priority < np.inf)
        self._update_max_pri(priority)
        index = self._v2i[value]
        self._tree[index] = (priority, value)
        self._update_up(index)

    def push_or_update(self, values, priorities):
        for value, priority in zip(values, priorities):
            if value in self._v2i:
                self.update_priority(value, priority)
            else:
                self.push(value, priority)

    def sample(self, ran):
        assert(self._curr_size > 0)
        assert(ran >= 0)
        assert(ran <= 1)
        search_index = 0
        search_val = ran * self.total_weight
        if len(self._tree) == 1:
            return self._tree[0][1], self._tree[0][0]
        else:
            return self._sample_down(search_index, search_val)

    ##############
    ### Helper ###
    ##############

    def _get_val(self, index):
        val = self._tree[index]
        if isinstance(val, tuple):
            return val[0]
        else:
            return val

    def _update_up(self, index):
        if index != 0:
            val_1 = self._get_val(index)
            if index % 2 == 0:
                val_2 = self._get_val(index - 1)
            else:
                val_2 = self._get_val(index + 1)
            val = val_1 + val_2
            index = (index - 1) // 2
            self._tree[index] = val
            self._update_up(index)

    def _sample_down(self, search_index, search_val):
        val = self._tree[search_index]
        if isinstance(val, tuple):
            return val[1], val[0]
        else:
            left_index = (search_index * 2) + 1
            right_index = left_index + 1
            left_val = self._get_val(left_index)
            if search_val <= left_val:
                return self._sample_down(left_index, search_val)
            else:
                return self._sample_down(right_index, search_val - left_val)

    def _update_max_pri(self, priority):
        self._max_priority = np.maximum(priority, self._max_priority)

if __name__ == '__main__':
    s = SumTree(10)
    for i in range(9):
        s.push(i, i + 1)
    print(s._tree)
    print(s._max_priority)
    s.update_priority(0, 11)
    print(s._tree)
    print(s._max_priority)
