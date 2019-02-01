import os
import copy
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.tf import networks
from gcg.tf import tf_utils
from gcg.misc import schedules
from gcg.misc import utils
from gcg.data.logger import logger


class GCGPolicy(object):
    def __init__(self, **kwargs):
        ### environment
        self._env_spec = kwargs['env_spec']
        self._obs_vec_keys = list(self._env_spec.observation_vec_spec.keys())
        self._action_keys = list(self._env_spec.action_spec.keys())
        self._goal_keys = list(self._env_spec.goal_spec.keys())
        self._output_keys = sorted([output['name'] for output in kwargs['outputs']])
        self._obs_im_shape = self._env_spec.observation_im_space.shape
        self._obs_im_num_channels = self._obs_im_shape[2]
        self._obs_im_dim = np.prod(self._obs_im_shape)
        self._obs_vec_dim = len(self._obs_vec_keys)
        self._action_dim = len(self._action_keys)
        self._goal_dim = len(self._goal_keys)

        ### model
        self._outputs = kwargs['outputs'] 
        self._action_selection_value = kwargs['action_selection_value']
        self._goals_to_input = kwargs['goals_to_input']
        for goal in self._goals_to_input:
            assert goal in self._goal_keys, '{0} goal is not in goal_keys'.format(goal)
        ### architecture
        self._inference_only = kwargs.get('inference_only', False)
        self._image_graph = kwargs['image_graph']
        self._observation_graph = kwargs['observation_graph']
        self._action_graph = kwargs['action_graph']
        self._rnn_graph = kwargs['rnn_graph']
        self._output_graph = kwargs['output_graph']
        ### scopes
        self._gcg_scope = 'gcg_scope'
        self._policy_scope = 'policy_scope'
        self._target_scope = 'target_scope'
        self._image_scope = 'image_scope'
        self._observation_scope = 'observation_scope'
        self._action_scope = 'action_scope'
        self._rnn_scope = 'rnn_scope'
        self._output_scope = 'output_scope'

        ### model horizons
        self._N = kwargs['N'] # number of returns to use (N-step)
        self._H = kwargs['H'] # action planning horizon for training
        self._gamma = kwargs['gamma'] # reward decay
        self._obs_history_len = kwargs['obs_history_len'] # how many previous observations to use

        ### target network
        self._use_target = kwargs['use_target']

        ### training
        self._optimizer = kwargs['optimizer']
        self._weight_decay = kwargs['weight_decay']
        self._lr_schedule = schedules.PiecewiseSchedule(**kwargs['lr_schedule'])
        self._grad_clip_norm = kwargs['grad_clip_norm']
        self._gpu_device = kwargs['gpu_device']
        self._gpu_frac = kwargs['gpu_frac']

        ### action selection and exploration
        assert (isinstance(self._env_spec.action_selection_space, Box))
        self._get_action_test = kwargs['get_action_test']
        self._get_action_target = kwargs['get_action_target']
        assert(self._get_action_target['type'] == 'random')
        self._exploration_stragies = [es['class'](self._env_spec, **es['params'])
                                      for es in kwargs['exploration_strategies']]

        ### setup the model
        self._seed = kwargs['seed']
        self._tf_debug = dict()
        self._tf_dict = self._graph_setup()

        ### logging
        self._log_stats = defaultdict(list)

        assert(self._N >= self._H)

        ### copy over target weights immediately (if there is a target)
        self.update_target()

    ##################
    ### Properties ###
    ##################

    @property
    def N(self):
        return self._N

    @property
    def gamma(self):
        return self._gamma

    @property
    def session(self):
        return self._tf_dict['sess']

    @property
    def obs_history_len(self):
        return self._obs_history_len

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_input_output_placeholders(self):
        with tf.variable_scope('input_output_placeholders'):
            ### policy inputs
            tf_obs_im_ph = tf.placeholder(tf.uint8, [None, self._obs_history_len, self._obs_im_dim], name='tf_obs_im_ph')
            tf_obs_vec_ph = tf.placeholder(tf.float32, [None, self._obs_history_len, self._obs_vec_dim], name='tf_obs_vec_ph')
            tf_actions_ph = tf.placeholder(tf.float32, [None, self._H, self._action_dim], name='tf_actions_ph')
            tf_dones_ph = tf.placeholder(tf.bool, [None, self._N + 1], name='tf_dones_ph')
            tf_goals_ph = tf.placeholder(tf.float32, [None, self._goal_dim], name='tf_goals_ph')
            ### policy outputs
            tf_rewards_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_rewards_ph')
            tf_weights_ph = tf.placeholder(tf.float32, [None], name='tf_weights_ph')
            ### target inputs
            tf_obs_im_target_ph = tf.placeholder(tf.uint8, [None, self._N + self._obs_history_len - 0, self._obs_im_dim], name='tf_obs_im_target_ph')
            tf_obs_vec_target_ph = tf.placeholder(tf.float32, [None, self._N + self._obs_history_len - 0, self._obs_vec_dim], name='tf_obs_vec_target_ph')
            tf_goals_target_ph = tf.placeholder(tf.float32, [None, self._N + 1, self._goal_dim], name='tf_goals_target_ph')
            ### episode timesteps
            tf_episode_timesteps_ph = tf.placeholder(tf.int32, [None], name='tf_episode_timesteps')

        return tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_dones_ph, tf_goals_ph, tf_rewards_ph, tf_weights_ph, \
               tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_goals_target_ph, tf_episode_timesteps_ph

    ### main graph

    def _graph_obs_to_lowd(self, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, is_training):
        with tf.name_scope('obs_to_lowd'):
            assert (self._obs_im_dim > 0)
            assert (self._image_graph is not None)
            assert (tf_obs_im_ph.dtype == tf.uint8)
            image_graph = copy.deepcopy(self._image_graph)

            ### CNN
            with tf.variable_scope(self._image_scope):
                ### whiten image
                tf_obs_im_whitened = (tf.cast(tf_obs_im_ph, tf.float32) - 128.) / 128.

                height, width, channels = self._obs_im_shape
                layer = tf.reshape(tf_obs_im_whitened, [-1, self._obs_history_len, height, width, channels])
                # [batch_size, hist_len, height, width, channels]

                layer = tf.transpose(layer, (0, 2, 3, 4, 1))
                # [batch_size, height, width, channels, hist_len]

                layer = tf.reshape(layer, [-1, height, width, channels * self._obs_history_len])
                # [batch_size, height, width, channels * hist_len]

                # convert to NCHW because faster
                layer = tf.transpose(layer, (0, 3, 1, 2))
                # [batch_size, channels * hist_len, height, width]

                layer = networks.convnn(layer,
                                        is_training=is_training,
                                        global_step_tensor=self.global_step,
                                        data_format='NCHW',
                                        **image_graph)
                layer = layers.flatten(layer)
                # pass through cnn to get [batch_size, ??]

            ### FCNN
            with tf.variable_scope(self._observation_scope):
                concat_list = [layer]
                if tf_obs_vec_ph.get_shape()[1].value > 0:
                    tf_obs_vec_whitened = self._graph_whiten(tf_obs_vec_ph, self._env_spec.observation_vec_spec)
                    concat_list.append(tf.reshape(tf_obs_vec_whitened, [-1, self._obs_history_len * self._obs_vec_dim]))
                if len(self._goals_to_input) > 0:
                    tf_goals_whitened = self._graph_whiten(tf_goals_ph, self._env_spec.goal_spec)
                    for goal_key in self._goals_to_input:
                        idx = self._goal_keys.index(goal_key)
                        concat_list.append(tf_goals_whitened[:, idx:idx+1])
                if len(concat_list) > 1:
                    layer = tf.concat(concat_list, axis=1)

                tf_obs_lowd = networks.fullyconnectednn(layer,
                                                        **self._observation_graph,
                                                        is_training=is_training,
                                                        global_step_tensor=self.global_step)

        return tf_obs_lowd

    def _graph_whiten(self, tf_x, spec):
        mean = []
        scale = []
        for k in spec.keys():
            spec_k = spec[k]
            if isinstance(spec_k, Box):
                mean.append(0.5 * (spec_k.high[0] + spec_k.low[0]))
                scale.append(0.5 * (spec_k.high[0] - spec_k.low[0]))
            elif isinstance(spec_k, Discrete):
                mean.append(float(spec_k.n) / 2.)
                scale.append(float(spec_k.n) / 2.)
            else:
                raise NotImplementedError
        assert (min(scale) > 1e-4)
        tf_x_whitened = (tf_x - np.array(mean, dtype=np.float32)) / np.array(scale, dtype=np.float32)

        return tf_x_whitened

    def _graph_inference(self, tf_obs_lowd, obs_vec, goals, tf_actions_ph, is_training):
        H = tf_actions_ph.get_shape()[1].value

        with tf.variable_scope(self._action_scope):
            ### whiten the actions
            action_space = self._env_spec.action_space
            action_mean = np.tile(0.5 * (action_space.low + action_space.high), (H, 1))
            action_scale = np.tile(action_space.high - action_space.low, (H, 1))
            tf_actions = (tf_actions_ph - action_mean) / action_scale

            self._action_graph.update({'output_dim': self._observation_graph['output_dim']})
            rnn_inputs = networks.fullyconnectednn(tf_actions,
                                                   **self._action_graph,
                                                   T=H,
                                                   is_training=is_training,
                                                   global_step_tensor=self.global_step)

        with tf.variable_scope(self._rnn_scope):
            rnn_outputs = networks.rnn(rnn_inputs,
                                       **self._rnn_graph,
                                       initial_state=tf_obs_lowd,
                                       is_training=is_training)

        with tf.variable_scope(self._output_scope):
            yhats = OrderedDict()
            pre_yhats = OrderedDict()
            with tf.variable_scope('yhats'):
                yhat_graph = copy.copy(self._output_graph)
                yhat_graph.update({'output_dim': 1})
                for output in self._outputs:
                    key = output['name']
                    if output.get('yhat'):
                        pre_yhat = networks.fullyconnectednn(rnn_outputs,
                                                             **yhat_graph,
                                                             T=H,
                                                             is_training=is_training,
                                                             scope=key,
                                                             global_step_tensor=self.global_step)[:, :, 0]
                        pre_yhats[key] = pre_yhat
                        yhats[key] = output['yhat'](pre_yhat=pre_yhat, obs_vec=obs_vec)
                        assert(len(yhats[key].get_shape()) == 2)

            bhats = OrderedDict()
            pre_bhats = OrderedDict()
            with tf.variable_scope('bhats'):
                for output in self._outputs:
                    key = output['name']
                    if output.get('bhat'):
                        bhat_graph = copy.copy(self._output_graph)
                        bhat_graph.update({'output_dim' : 1})
                        pre_bhat = networks.fullyconnectednn(rnn_outputs,
                                                             **bhat_graph,
                                                             T=H,
                                                             is_training=is_training,
                                                             scope=key,
                                                             global_step_tensor=self.global_step)[:, :, 0]
                        pre_bhats[key] = pre_bhat
                        bhats[key] = output['bhat'](pre_bhat=pre_bhat, obs_vec=obs_vec)
                        assert(len(bhats[key].get_shape()) == 2)

        values = self._graph_calculate_value(yhats, bhats, goals, H)
        return values, yhats, bhats, pre_yhats, pre_bhats

    def _graph_calculate_value(self, yhats, bhats, goals, H):
        values = OrderedDict()
        for output in self._outputs:
            if output.get('value'):
                value = output['value'](yhats=yhats, bhats=bhats, goals=goals, gamma=self._gamma)
                assert(len(value.get_shape()) == 1)
                values[output['name']] = tf.expand_dims(value, 1)

        return values

    ### action selection / target maximization

    def _graph_get_action(self, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals, get_action_params,
                          scope_select, reuse_select, scope_eval, reuse_eval, tf_episode_timesteps_ph):
        get_action_type = get_action_params['type']

        if get_action_type == 'random':
            return self._graph_get_action_random(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals, get_action_params,
                                                 scope_select, reuse_select, scope_eval, reuse_eval, tf_episode_timesteps_ph)
        elif get_action_type == 'cem':
            return self._graph_get_action_cem(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals,get_action_params,
                                              scope_select, reuse_select, scope_eval, reuse_eval, tf_episode_timesteps_ph)
        else:
            raise NotImplementedError

    def _graph_get_action_random(self, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals, get_action_params,
                                 scope_select, reuse_select, scope_eval, reuse_eval, tf_episode_timesteps_ph):
        H = get_action_params['H']
        N = self._N
        h_select = get_action_params.get('h_select', 0)
        assert (H <= N)
        assert (h_select < H)
        num_obs = tf.shape(tf_obs_im_ph)[0]

        random_params = get_action_params['random']
        K = random_params['K']
        get_action_selection = random_params.get('selection', 'argmax')

        ### create actions
        tf_actions_all = self._graph_generate_random_actions([K, H])

        ### process to lowd
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph,
                                                         is_training=False)
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_obs_lowd_eval = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph,
                                                       is_training=False)
        ### tile
        tf_actions_all = tf.tile(tf_actions_all, (num_obs, 1, 1))
        tf_obs_lowd_repeat_select = tf_utils.repeat_2d(tf_obs_lowd_select, K, 0)
        tf_obs_lowd_repeat_eval = tf_utils.repeat_2d(tf_obs_lowd_eval, K, 0)
        ### inference to get values
        with tf.variable_scope(scope_select, reuse=reuse_select):
            values_all_select, yhats_all_select, bhats_all_select, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_select, obs_vec, goals, tf_actions_all,
                                      is_training=False)  # [num_obs*K, H]
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            values_all_eval, yhats_all_eval, bhats_all_eval, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_eval, obs_vec, goals, tf_actions_all,
                                      is_training=False)

        actions = OrderedDict()
        for i, key in enumerate(self._action_keys):
            actions[key] = tf_actions_all[:, :, i]

        act_inputs = OrderedDict()
        for key in obs_vec:
            act_inputs[key] = tf_utils.repeat_2d(obs_vec[key], K, 0)

        act_goals = OrderedDict()
        for key in goals:
            act_goals[key] = tf_utils.repeat_2d(goals[key], K, 0)

        tf_values_select = self._get_action_value(actions, values_all_select, yhats_all_select, bhats_all_select, act_inputs, act_goals)
        tf_values_eval = self._get_action_value(actions, values_all_eval, yhats_all_eval, bhats_all_eval, act_inputs, act_goals)

        ### get_action based on select (policy)
        tf_values_select = tf.reshape(tf_values_select, (num_obs, K))  # [num_obs, K]
        if get_action_selection == 'argmax':
            tf_select_chosen = tf.argmax(tf_values_select, axis=1)
        else:
            raise NotImplementedError
        tf_select_mask = tf.expand_dims(tf.one_hot(tf_select_chosen, depth=K), axis=2)  # [num_obs, K, 1]
        tf_get_action = tf.reduce_sum(
            tf_select_mask * tf.reshape(tf_actions_all, (num_obs, K, H, self._action_dim))[:, :, h_select, :],
            reduction_indices=1)  # [num_obs, action_dim]
        ### get_action_value based on eval (target)
        if get_action_selection == 'argmax':
            tf_values_eval = tf.reshape(tf_values_eval, (num_obs, K, 1))  # [num_obs, K, 1]
            tf_get_action_value = tf.reduce_sum(tf_select_mask * tf_values_eval, axis=(1, 2))
        else:
            raise NotImplementedError

        action_values = OrderedDict()
        for key in values_all_eval.keys():
            action_values[key] = tf.reduce_sum(tf.reshape(values_all_eval[key], (num_obs, K, H)) * tf_select_mask,
                                               axis=1)

        action_yhats = OrderedDict()
        for key in yhats_all_eval.keys():
            action_yhats[key] = tf.reduce_sum(tf.reshape(yhats_all_eval[key], (num_obs, K, H)) * tf_select_mask, axis=1)
            yhats_all_eval[key] = tf.reshape(yhats_all_eval[key], (num_obs, K, H))

        action_bhats = OrderedDict()
        for key in bhats_all_eval.keys():
            action_bhats[key] = tf.reduce_sum(tf.reshape(bhats_all_eval[key], (num_obs, K, H)) * tf_select_mask,
                                              axis=1)
            bhats_all_eval[key] = tf.reshape(bhats_all_eval[key], (num_obs, K, H))

        ### check shapes
        assert (tf_get_action.get_shape()[1].value == self._action_dim)

        tf_get_action_reset_ops = []

        # action selection
        # target values
        # action reset (e.g., for CEM)
        # extra info (for debugging)
        return tf_get_action, tf_get_action_value, \
               action_values, action_yhats, action_bhats, \
               tf_get_action_reset_ops, \
               tf_actions_all, tf_values_select, yhats_all_eval, bhats_all_eval

    def _graph_get_action_cem(self, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals, get_action_params,
                              scope_select, reuse_select, scope_eval, reuse_eval, tf_episode_timesteps_ph):
        H = get_action_params['H']
        N = self._N
        h_select = get_action_params.get('h_select', 0)
        assert (H <= N)
        assert (h_select < H)
        get_action_type = get_action_params['type']
        num_obs = tf.shape(tf_obs_im_ph)[0]
        aspace_select = self._env_spec.action_selection_space

        assert (get_action_type == 'cem')

        cem_params = get_action_params['cem']
        M_init = cem_params['M_init']
        M = cem_params['M']
        K = cem_params['K']
        itrs = cem_params['itrs']
        eps = cem_params['eps']

        control_dependencies = []
        control_dependencies += [tf.assert_equal(num_obs, 1)]
        with tf.control_dependencies(control_dependencies):
            ### process to lowd
            with tf.variable_scope(scope_select, reuse=reuse_select):
                tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, is_training=False)

            ### initialize CEM
            Ms = [M_init] + [M] * (itrs - 1)
            Ks = [K] * (itrs - 1) + [1]
            tf_obs_lowd_select_repeats = [tf.tile(tf_obs_lowd_select, (M_init, 1))] + \
                                         [tf.tile(tf_obs_lowd_select, (M, 1))] * (itrs - 1)
            distribution = tf.contrib.distributions.Uniform(list(aspace_select.low) * H,
                                                            list(aspace_select.high) * H)

            ### run CEM
            for M_i, K_i, tf_obs_lowd_select_repeat_i in zip(Ms, Ks, tf_obs_lowd_select_repeats):
                tf_flat_actions_preclip = distribution.sample((M_i,))
                tf_flat_actions = tf.clip_by_value(
                    tf_flat_actions_preclip,
                    np.array([list(aspace_select.low) * H] * M_i, dtype=np.float32),
                    np.array([list(aspace_select.high) * H] * M_i, dtype=np.float32))
                tf_actions = tf.reshape(tf_flat_actions, (M_i, H, aspace_select.flat_dim))

                with tf.variable_scope(scope_select, reuse=True):
                    values_all_select, yhats_all_select, bhats_all_select, _, _ = \
                        self._graph_inference(tf_obs_lowd_select_repeat_i, obs_vec, goals, tf_actions,
                                              is_training=False)  # [num_obs*M_i, H]

                tf_values_select = self._get_action_value(tf_actions, values_all_select, yhats_all_select, bhats_all_select, obs_vec, goals)

                ### get top k
                _, top_indices = tf.nn.top_k(tf_values_select, k=K_i)
                top_controls = tf.gather(tf_flat_actions, indices=top_indices)

                ### set new distribution based on top k
                mean = tf.reduce_mean(top_controls, axis=0)
                covar = tf.matmul(tf.transpose(top_controls), top_controls) / float(K)
                sigma = covar + eps * tf.eye(H * aspace_select.flat_dim)

                distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(
                    loc=mean,
                    covariance_matrix=sigma
                )

            ### eval mean of final distribution

            tf_get_actions = tf.expand_dims(tf.reshape(distribution.mean(), (H, aspace_select.flat_dim)), 0)
            tf_get_action = tf_get_actions[:, h_select, :]

            with tf.variable_scope(scope_eval, reuse=reuse_eval):
                tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, is_training=False)
                values_all_eval, yhats_all_eval, bhats_all_eval, _, _ = \
                    self._graph_inference(tf_obs_lowd_select, obs_vec, goals, tf_get_actions,
                                          is_training=False)
            tf_get_action_value = self._get_action_value(tf_get_actions, values_all_eval, yhats_all_eval, bhats_all_eval, obs_vec, goals)

            # can't use CEM for target computation, so no need
            action_values = None
            action_yhats = None
            action_bhats = None

            ### check shapes
            assert (tf_get_action.get_shape()[1].value == self._action_dim)

            tf_get_action_reset_ops = []

            ### make them all have num_obs=1 as the first dimension
            tf_values_select = tf.expand_dims(tf_values_select, 0)
            for key in yhats_all_select.keys():
                yhats_all_select[key] = tf.expand_dims(yhats_all_select[key], 0)
            for key in bhats_all_select.keys():
                bhats_all_select[key] = tf.expand_dims(bhats_all_select[key], 0)

            # action selection
            # target values
            # action reset (e.g., for CEM)
            # extra info (for debugging)
            return tf_get_action, tf_get_action_value, \
                   action_values, action_yhats, action_bhats, \
                   tf_get_action_reset_ops, \
                   tf_actions, tf_values_select, yhats_all_select, bhats_all_select # select b/c eval is just one, the mean

    def _graph_generate_random_actions(self, shape):
        action_lb = np.tile(utils.multiple_expand_dims(self._env_spec.action_selection_space.low, [0] * len(shape)),
                            shape + [1])
        action_ub = np.tile(utils.multiple_expand_dims(self._env_spec.action_selection_space.high, [0] * len(shape)),
                            shape + [1])
        tf_actions = (action_ub - action_lb) * tf.random_uniform(shape + [self._action_dim]) + action_lb

        return tf_actions

    def _get_action_value(self, actions, values, yhats, bhats, obs_vec, goals):
        if not isinstance(actions, OrderedDict):
            d_actions = OrderedDict()
            for i, key in enumerate(self._action_keys):
                d_actions[key] = actions[:, :, i]
            actions = d_actions

        value = self._action_selection_value(actions=actions,
                                             yhats=yhats,
                                             bhats=bhats,
                                             values=values,
                                             goals=goals)

        value = tf.reduce_sum(value, axis=1)
        assert(len(value.shape) == 1)
        return value

    def _graph_get_target_values(self, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals, get_target_params,
                                 scope_select, reuse_select, scope_eval, reuse_eval, tf_episode_timesteps_ph):
        get_target_params = copy.deepcopy(get_target_params)
        assert (get_target_params['type'] == 'random')

        # pre just means before reorganization. it doesn't refer to pre_yhat or pre_bhat
        _, _, pre_target_values, pre_target_yhats, pre_target_bhats, _, _, _, _, _ = \
            self._graph_get_action(tf_obs_im_ph,
                                   tf_obs_vec_ph,
                                   tf_goals_ph,
                                   obs_vec,
                                   goals,
                                   get_target_params,
                                   scope_select,
                                   reuse_select,
                                   scope_eval,
                                   reuse_eval,
                                   tf_episode_timesteps_ph)

        H_target = get_target_params['H']
        target_values = OrderedDict()
        for key in pre_target_values.keys():
            target_values[key] = tf.transpose(tf.reshape(pre_target_values[key], (self._N, -1)), (1, 0))

        target_yhats = OrderedDict()
        for key in pre_target_yhats.keys():
            target_yhats[key] = tf.transpose(tf.reshape(pre_target_yhats[key], (self._N, -1, H_target)), (1, 0, 2))

        target_bhats = OrderedDict()
        for key in pre_target_bhats.keys():
            target_bhats[key] = tf.transpose(tf.reshape(pre_target_bhats[key], (self._N, -1, H_target)), (1, 0, 2))

        return target_values, target_yhats, target_bhats

    ### training

    def _graph_cost(self, values, yhats, bhats, pre_yhats, pre_bhats, obs_vec, goals, future_goals,
                    actions, tf_obs_vec_target_ph, tf_rewards_ph, tf_dones_ph,
                    target_obs_vec, target_values, target_yhats, target_bhats,
                    tf_weights_ph, N=None):
        N = self._N if N is None else N
        assert(tf_rewards_ph.get_shape()[1].value == N)
        assert(tf_dones_ph.get_shape()[1].value == N+1)

        control_dependencies = []
        costs = dict()
        costs_yhat = dict()
        costs_bhat = dict()
        accs_yhat = dict()

        ### mask
        tf_weights = tf.expand_dims(tf_weights_ph, axis=1)
        tf_dones = tf.cast(tf_dones_ph, tf.float32)
        lengths = tf.reduce_sum(1 - tf_dones, axis=1)
        control_dependencies.append(tf.assert_greater(lengths, 0., name='length_assert'))
        
        clip_mask = tf.sequence_mask(
            tf.cast(lengths, tf.int32),
            maxlen=self._H,
            dtype=tf.float32)
        clip_mask /= tf.reduce_sum(clip_mask)
        
        no_clip_mask = tf.ones(tf.shape(clip_mask), dtype=tf.float32)
        no_clip_mask /= tf.reduce_sum(no_clip_mask)

        clip_mask *= tf_weights
        no_clip_mask *= tf_weights

        assert(len(clip_mask.get_shape()) == len(tf_weights.get_shape()))
        assert(len(no_clip_mask.get_shape()) == len(tf_weights.get_shape()))

        yhat_labels = OrderedDict()
        bhat_labels = OrderedDict()
        
        for output in self._outputs:
            key = output['name']
            if output['clip_with_done']:
                mask = clip_mask
            else:
                mask = no_clip_mask
           
            cost = 0.
            cost_yhat = tf.constant(0.0)
            cost_bhat = tf.constant(0.0)
            acc_yhat = None

            if output.get('yhat'):
                yhat = yhats[output['name']]
                pre_yhat = pre_yhats[output['name']]
                yhat_label = output['yhat_label'](rewards=tf_rewards_ph,
                                                  dones=tf_dones,
                                                  goals=goals,
                                                  future_goals=future_goals,
                                                  target_obs_vec=target_obs_vec,
                                                  gamma=self._gamma)
                yhat_labels[output['name']] = yhat_label
                assert(yhat_label.shape.as_list() == yhat.shape.as_list())
                yhat_loss = output['yhat_loss']
                yhat_loss_use_pre = output.get('yhat_loss_use_pre', False)
                yhat_loss_xentropy_posweight = output.get('yhat_loss_xentropy_posweight', 1)
                yhat_scale = output.get('yhat_scale', 1.0)
                yhat_loss_weight = output.get('yhat_loss_weight', 1.0)
                cost_yhat, acc_yhat, cost_dep = self._graph_sub_cost(yhat, pre_yhat, yhat_loss_use_pre, yhat_label,
                                                                     mask, yhat_loss, yhat_scale, yhat_loss_weight,
                                                                     xentropy_posweight=yhat_loss_xentropy_posweight)
                cost += cost_yhat
                control_dependencies += cost_dep
           
            if output.get('bhat'):
                bhat = bhats[output['name']]
                pre_bhat = pre_bhats[output['name']]
                bhat_label = output['bhat_label'](rewards=tf_rewards_ph,
                                                  dones=tf_dones,
                                                  goals=goals,
                                                  target_obs_vec=target_obs_vec,
                                                  gamma=self._gamma,
                                                  future_goals=future_goals,
                                                  target_yhats=target_yhats,
                                                  target_bhats=target_bhats,
                                                  target_values=target_values)
                bhat_labels[output['name']] = bhat_label
                assert(bhat_label.shape.as_list() ==  bhat.shape.as_list())
                bhat_loss = output['bhat_loss']
                bhat_loss_use_pre = output.get('bhat_loss_use_pre', False)
                bhat_loss_xentropy_posweight = output.get('bhat_loss_xentropy_posweight', 1)
                bhat_scale = output.get('bhat_scale', 1.0)
                bhat_loss_weight = output.get('bhat_loss_weight', 1.0)
                cost_bhat, _, cost_dep = self._graph_sub_cost(bhat, pre_bhat, bhat_loss_use_pre, bhat_label,
                                                              mask, bhat_loss, bhat_scale, bhat_loss_weight,
                                                              xentropy_posweight=bhat_loss_xentropy_posweight)
                cost += cost_bhat
                control_dependencies += cost_dep

            costs[key] = cost
            costs_yhat[key] = cost_yhat
            costs_bhat[key] = cost_bhat
            if acc_yhat is not None:
                accs_yhat[key] = acc_yhat

        value_labels = self._graph_calculate_value(yhat_labels, bhat_labels, goals, self._H)

        rew = self._get_action_value(actions, values, yhats, bhats, obs_vec, goals)
        rew_label = self._get_action_value(actions, value_labels, yhat_labels, bhat_labels, obs_vec, goals)

        rew_errors = tf.abs(rew - rew_label)

        with tf.control_dependencies(control_dependencies):
            total_cost = tf.reduce_sum(list(costs.values()))

            ### weight decay
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(reg_losses) > 0:
                num_trainable_vars = float(np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()]))
                weight_decay = (self._weight_decay / num_trainable_vars) * tf.add_n(reg_losses)
            else:
                weight_decay = 0
            total_cost_with_reg = total_cost + weight_decay

        return total_cost_with_reg, total_cost, costs, costs_yhat, costs_bhat, accs_yhat, rew_errors

    def _graph_sub_cost(self, preds, pre_preds, use_pre, labels, mask, loss, scale, loss_weight, **kwargs):
        control_dependencies = []
        acc = None
        if loss == 'mse':
            assert(not use_pre)
            assert(len(preds.get_shape()) == len(labels.get_shape()))
            cost = 0.5 * tf.square(preds - labels) / scale
        elif loss == 'huber':
            # Used implementation similar to tf github to avoid gradient issues
            assert(not use_pre)
            assert(len(preds.get_shape()) == len(labels.get_shape()))
            delta = 1.0 * scale
            abs_diff = tf.abs(preds - labels)
            quadratic = tf.minimum(abs_diff, delta)
            linear = (abs_diff - quadratic)
            cost = (0.5 * quadratic**2 + delta * linear) / scale
        elif loss == 'xentropy':
            assert(use_pre)
            assert(len(pre_preds.get_shape()) == len(labels.get_shape()))
            labels /= scale
            preds /= scale
            control_dependencies += [tf.assert_greater_equal(labels, 0., name='cost_assert_2')]
            control_dependencies += [tf.assert_less_equal(labels, 1., name='cost_assert_3', summarize=1000)]
            control_dependencies += [tf.assert_greater_equal(preds, 0., name='cost_assert_4')]
            control_dependencies += [tf.assert_less_equal(preds, 1., name='cost_assert_5')]
            xentropy_posweight = kwargs.get('xentropy_posweight', 1)
            cost = tf.nn.weighted_cross_entropy_with_logits(logits=pre_preds, targets=labels,
                                                            pos_weight=xentropy_posweight)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds > 0.5, labels > 0.5), tf.float32))
        else:
            raise NotImplementedError
        assert(len(cost.get_shape()) == len((mask * loss_weight).get_shape()))
        cost = tf.reduce_sum(cost * mask * loss_weight)
        return cost, acc, control_dependencies

    def _graph_optimize(self, tf_cost, tf_policy_vars):
        tf_lr_ph = tf.placeholder(tf.float32, (), name="learning_rate")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        num_parameters = 0
        with tf.control_dependencies(update_ops):
            if self._optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr_ph, epsilon=1e-4)
            elif self._optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf_lr_ph)
            else:
                raise NotImplementedError
            gradients = optimizer.compute_gradients(tf_cost, var_list=tf_policy_vars)
            for i, (grad, var) in enumerate(gradients):
                num_parameters += int(np.prod(var.get_shape().as_list()))
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self._grad_clip_norm), var)
            tf_opt = optimizer.apply_gradients(gradients, global_step=self.global_step)
        logger.debug('Number of parameters: {0:e}'.format(float(num_parameters)))
        return tf_opt, tf_lr_ph

    def _graph_init_vars(self, tf_sess):
        tf_sess.run([tf.global_variables_initializer()])

    def _graph_setup_savers(self, tf_inference_vars, tf_all_vars, inference_only):
        savers_dict = dict()

        def filter_policy_vars(vars, must_contain):
            return [v for v in vars if must_contain in v.name]

        name_and_vars= [('inference', tf_inference_vars)]
        if not inference_only:
            name_and_vars.append(('train', tf_all_vars))

        savers_vars = dict()
        for name, vars in name_and_vars:
            savers_vars[name] = vars
            savers_vars[name + '_image'] = filter_policy_vars(vars, self._image_scope)
            savers_vars[name + '_observation'] = filter_policy_vars(vars, self._observation_scope)
            savers_vars[name + '_action'] = filter_policy_vars(vars, self._action_scope)
            savers_vars[name + '_rnn'] = filter_policy_vars(vars, self._rnn_scope)
            savers_vars[name + '_output'] = filter_policy_vars(vars, self._output_scope)

        for name, vars in savers_vars.items():
            assert name not in savers_dict.keys()
            savers_dict[name] = tf.train.Saver(vars, max_to_keep=None)

        return savers_dict

    ### high-level

    def _graph_setup_policy(self, obs_vec, goals, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, tf_actions_ph,
                            reuse=False, is_training=True):
        ### policy
        with tf.variable_scope(self._policy_scope, reuse=reuse):
            ### process obs to lowd
            tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, is_training=is_training)
            ### create training policy
            values, yhats, bhats, pre_yhats, pre_bhats = \
                self._graph_inference(tf_obs_lowd, obs_vec, goals, tf_actions_ph[:, :self._H, :], is_training=is_training)

        return values, yhats, bhats, pre_yhats, pre_bhats

    def _graph_setup_action_selection(self, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals,
                                      tf_episode_timesteps_ph):
        ### action selection
        tf_get_action, tf_get_action_value, _, _, _, tf_get_action_reset_ops,\
        tf_actions_all, tf_values_select, tf_yhats_all_eval, tf_bhats_all_eval = \
            self._graph_get_action(tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, obs_vec, goals, self._get_action_test,
                                   self._policy_scope, True, self._policy_scope, True,
                                   tf_episode_timesteps_ph)

        return tf_get_action, tf_get_action_value, tf_get_action_reset_ops, tf_actions_all, tf_values_select,\
               tf_yhats_all_eval, tf_bhats_all_eval

    def _graph_setup_target(self, tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_goals_target_ph,
                            target_obs_vec, goals, tf_policy_vars, reuse_eval=False):
        ### create target network
        if self._use_target:
            ### action selection
            tf_obs_im_target_ph_packed = tf.concat([tf_obs_im_target_ph[:, h - self._obs_history_len:h, :]
                                                    for h in range(self._obs_history_len + 1,
                                                                   self._obs_history_len + self._N + 1)],
                                                   0)
            tf_obs_vec_target_ph_packed = tf.concat([tf_obs_vec_target_ph[:, h - self._obs_history_len:h, :]
                                                     for h in range(self._obs_history_len + 1,
                                                                    self._obs_history_len + self._N + 1)],
                                                    0)
            tf_goals_target_ph_packed = tf_utils.repeat_2d(tf_goals_target_ph[:, 0, :], self._N, 0)

            target_values, target_yhats, target_bhats = \
                self._graph_get_target_values(tf_obs_im_target_ph_packed,
                                              tf_obs_vec_target_ph_packed,
                                              tf_goals_target_ph_packed,
                                              target_obs_vec, goals,
                                              self._get_action_target,
                                              scope_select=self._policy_scope,
                                              reuse_select=True,
                                              scope_eval=self._target_scope,
                                              reuse_eval=reuse_eval or (self._target_scope == self._policy_scope),
                                              tf_episode_timesteps_ph=None)


        else:
            target_values = OrderedDict()
            target_yhats = OrderedDict()
            target_bhats = OrderedDict()
        ### update target network
        if self._use_target:
            tf_target_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope='{0}/{1}'.format(self._gcg_scope, self._target_scope)),
                                                      key=lambda v: v.name)
            assert (len(tf_policy_vars) > 0)
            assert (len(tf_policy_vars) == len(tf_target_vars))
            tf_update_target_fn = []
            for var, var_target in zip(tf_policy_vars, tf_target_vars):
                assert (var.name.replace(self._policy_scope, '') == var_target.name.replace(self._target_scope, ''))
                tf_update_target_fn.append(var_target.assign(var))
            tf_update_target_fn = tf.group(*tf_update_target_fn)
        else:
            tf_target_vars = None
            tf_update_target_fn = None

        return target_values, target_yhats, target_bhats, tf_target_vars, tf_update_target_fn

    def _graph_setup_ordered_dicts(self, tf_obs_vec, tf_goals, tf_goals_target, tf_actions):
        obs_vec = OrderedDict()
        for i, key in enumerate(self._obs_vec_keys):
            obs_vec[key] = tf_obs_vec[:, -1, i:i + 1]

        goals = OrderedDict()
        for i, key in enumerate(self._goal_keys):
            goals[key] = tf_goals[:, i:i + 1]

        future_goals = OrderedDict()
        for i, key in enumerate(self._goal_keys):
            future_goals[key] = tf_goals_target[:, :-1, i]

        actions = OrderedDict()
        for i, key in enumerate(self._action_keys):
            actions[key] = tf_actions[:, :, i]

        return obs_vec, goals, future_goals, actions

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = tf_utils.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            tf.set_random_seed(self._seed)
            with tf.variable_scope(self._gcg_scope):
                ### create input output placeholders
                tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_dones_ph, tf_goals_ph, tf_rewards_ph, tf_weights_ph, \
                tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_goals_target_ph, tf_episode_timesteps_ph = \
                    self._graph_input_output_placeholders()
                self.global_step = tf.Variable(0, trainable=False, name='{0}/global_step'.format(self._gcg_scope))

                obs_vec, goals, future_goals, actions = self._graph_setup_ordered_dicts(tf_obs_vec_ph,
                                                                                        tf_goals_ph,
                                                                                        tf_goals_target_ph,
                                                                                        tf_actions_ph)

                ### setup policy
                values, yhats, bhats, pre_yhats, pre_bhats = \
                    self._graph_setup_policy(obs_vec, goals, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, tf_actions_ph)
                action_value = self._get_action_value(tf_actions_ph, values, yhats, bhats, obs_vec, goals)

                ### get action
                tf_get_action, tf_get_action_value,\
                tf_get_action_reset_ops,\
                tf_get_action_all, tf_get_action_value_all, tf_get_action_yhats_all, tf_get_action_bhats_all = \
                    self._graph_setup_action_selection(tf_obs_im_ph, tf_obs_vec_ph,
                                                       tf_goals_ph, obs_vec, goals, tf_episode_timesteps_ph)

                ### get policy variables
                tf_inference_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                             scope='{0}/{1}'.format(self._gcg_scope, self._policy_scope)),
                                           key=lambda v: v.name)
                tf_trainable_policy_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                    scope='{0}/{1}'.format(self._gcg_scope, self._policy_scope)),
                                                  key=lambda v: v.name)

                if not self._inference_only:
                    assert (len(tf_trainable_policy_vars) > 0)

                    target_obs_vec = OrderedDict()
                    for i, key in enumerate(self._obs_vec_keys):
                        target_obs_vec[key] = tf_obs_vec_target_ph[:, -self._N:, i]

                    ### setup target
                    target_values, target_yhats, target_bhats, tf_target_vars, tf_update_target_fn = \
                        self._graph_setup_target(tf_obs_im_target_ph, tf_obs_vec_target_ph,
                                                 tf_goals_target_ph, target_obs_vec, goals, tf_inference_vars)

                    ### optimization
                    tf_cost_with_reg, tf_cost, tf_costs, tf_costs_yhat, tf_costs_bhat, tf_accs_yhat, tf_rew_errors = \
                        self._graph_cost(values, yhats, bhats, pre_yhats, pre_bhats, obs_vec, goals, future_goals,
                                         actions, tf_obs_vec_target_ph, tf_rewards_ph, tf_dones_ph,
                                         target_obs_vec, target_values, target_yhats, target_bhats, tf_weights_ph)
                    tf_opt, tf_lr_ph = self._graph_optimize(tf_cost_with_reg, tf_trainable_policy_vars)
                else:
                    tf_cost_with_reg, tf_cost, tf_costs, tf_costs_yhat, tf_costs_bhat, tf_accs_yhat = \
                        None, None, dict(), dict(), dict(), dict()
                    tf_target_vars = tf_update_target_fn = tf_cost = tf_opt = tf_lr_ph = tf_rew_errors = None

            ### savers
            tf_all_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._gcg_scope),
                                 key=lambda v: v.name)
            tf_savers_dict = self._graph_setup_savers(tf_inference_vars, tf_all_vars, self._inference_only)

            ### initialize
            self._graph_init_vars(tf_sess)

        ### what to return
        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_im_ph': tf_obs_im_ph,
            'obs_vec_ph': tf_obs_vec_ph,
            'goals_ph': tf_goals_ph,
            'actions_ph': tf_actions_ph,
            'dones_ph': tf_dones_ph,
            'rewards_ph': tf_rewards_ph,
            'weights_ph': tf_weights_ph,
            'obs_im_target_ph': tf_obs_im_target_ph,
            'obs_vec_target_ph': tf_obs_vec_target_ph,
            'goals_target_ph': tf_goals_target_ph,
            'episode_timesteps_ph': tf_episode_timesteps_ph,
            'yhats': yhats,
            'bhats': bhats,
            'values': values,
            'action_value': action_value,
            'get_action': tf_get_action,
            'get_action_value': tf_get_action_value,
            'get_action_reset_ops': tf_get_action_reset_ops,
            'get_action_all': tf_get_action_all,
            'get_action_value_all': tf_get_action_value_all,
            'get_action_yhats_all': tf_get_action_yhats_all,
            'get_action_bhats_all': tf_get_action_bhats_all,
            'update_target_fn': tf_update_target_fn,
            'cost_with_reg': tf_cost_with_reg,
            'cost': tf_cost,
            'costs': tf_costs,
            'costs_yhat': tf_costs_yhat,
            'costs_bhat': tf_costs_bhat,
            'accs_yhat': tf_accs_yhat,
            'rew_errors': tf_rew_errors,
            'opt': tf_opt,
            'lr_ph': tf_lr_ph,
            'savers_dict': tf_savers_dict
        }

    ################
    ### Training ###
    ################

    def update_target(self):
        if self._use_target and not self._inference_only and self._tf_dict['update_target_fn']:
            self._tf_dict['sess'].run(self._tf_dict['update_target_fn'])

    def train_step(self, step, steps, observations, goals, actions, rewards, dones, weights):
        """
        :param steps: [batch_size, N+1]
        :param observations_im: [batch_size, N+1 + obs_history_len-1, obs_im_dim]
        :param observations_vec: [batch_size, N+1 + obs_history_len-1, obs_vec_dim]
        :param goals: [batch_size, N+1, goal_dim]
        :param actions: [batch_size, H+1, action_dim]
        :param rewards: [batch_size, N+1]
        :param dones: [batch_size, N+1]
        """
        for v in (steps, observations[0], observations[1], goals, actions, rewards, dones, weights):
            assert not np.isnan(v).any(), '{0} is nan in train_step'.format(v)

        observations_im, observations_vec = observations
        feed_dict = {
            ### parameters
            self._tf_dict['lr_ph']: self._lr_schedule.value(step),
            ### policy
            self._tf_dict['obs_im_ph']: observations_im[:, :self._obs_history_len, :],
            self._tf_dict['obs_vec_ph']: observations_vec[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions[:, :self._H, :],
            self._tf_dict['dones_ph']: dones[:, :self._N+1],
            self._tf_dict['goals_ph']: goals[:, 0],
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
            self._tf_dict['weights_ph']: weights,
            self._tf_dict['obs_vec_target_ph']: observations_vec,
            self._tf_dict['goals_target_ph']: goals,
        }
        if self._use_target:
            feed_dict[self._tf_dict['obs_im_target_ph']] = observations_im

        rew_errors, cost_with_reg, cost, costs, costs_yhat, costs_bhat, accs_yhat, _ = \
            self._tf_dict['sess'].run([self._tf_dict['rew_errors'],
                                       self._tf_dict['cost_with_reg'],
                                       self._tf_dict['cost'],
                                       self._tf_dict['costs'],
                                       self._tf_dict['costs_yhat'],
                                       self._tf_dict['costs_bhat'],
                                       self._tf_dict['accs_yhat'],
                                       self._tf_dict['opt']],
                                      feed_dict=feed_dict)
        assert(np.isfinite(cost))

        self._log_stats['cost'].append(cost)
        self._log_stats['cost with reg'].append(cost_with_reg)
        self._log_stats['cost fraction reg'].append((cost_with_reg - cost) / cost_with_reg)
        for output in self._outputs:
            name = output['name']
            if name in costs:
                self._log_stats['{0} cost'.format(name)].append(costs[name])
            if name in costs_yhat:
                self._log_stats['{0} cost yhat'.format(name)].append(costs_yhat[name])
            if name in accs_yhat:
                self._log_stats['{0} acc yhat'.format(name)].append(accs_yhat[name])
            if name in costs_bhat:
                self._log_stats['{0} cost bhat'.format(name)].append(costs_bhat[name])

        return rew_errors

    def eval_holdout(self, step, steps, observations, goals, actions, rewards, dones):
        for v in (steps, observations[0], observations[1], goals, actions, rewards, dones):
            assert not np.isnan(v).any(), '{0} is nan in eval_holdout'.format(v)

        observations_im, observations_vec = observations
        feed_dict = {
            ### policy
            self._tf_dict['obs_im_ph']: observations_im[:, :self._obs_history_len, :],
            self._tf_dict['obs_vec_ph']: observations_vec[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions[:, :self._H, :],
            self._tf_dict['dones_ph']: dones[:, :self._N+1],
            self._tf_dict['goals_ph']: goals[:, 0],
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
            self._tf_dict['weights_ph']: np.ones(len(steps), dtype=np.float32),
            self._tf_dict['obs_vec_target_ph']: observations_vec,
            self._tf_dict['goals_target_ph']: goals,
        }
        if self._use_target:
            feed_dict[self._tf_dict['obs_im_target_ph']] = observations_im

        cost_with_reg, cost, costs, costs_yhat, costs_bhat, accs_yhat = \
            self._tf_dict['sess'].run([self._tf_dict['cost_with_reg'],
                                       self._tf_dict['cost'],
                                       self._tf_dict['costs'],
                                       self._tf_dict['costs_yhat'],
                                       self._tf_dict['costs_bhat'],
                                       self._tf_dict['accs_yhat']],
                                      feed_dict=feed_dict)

        self._log_stats['cost holdout'].append(cost)
        self._log_stats['cost with reg holdout'].append(cost_with_reg)
        for output in self._outputs:
            name = output['name']
            if name in costs:
                self._log_stats['{0} cost holdout'.format(name)].append(costs[name])
            if name in costs_yhat:
                self._log_stats['{0} cost yhat holdout'.format(name)].append(costs_yhat[name])
            if name in accs_yhat:
                self._log_stats['{0} acc yhat holdout'.format(name)].append(accs_yhat[name])
            if name in costs_bhat:
                self._log_stats['{0} cost bhat holdout'.format(name)].append(costs_bhat[name])

        return cost_with_reg, cost, costs, costs_yhat, costs_bhat, accs_yhat

    def reset_weights(self):
        tf_sess = self._tf_dict['sess']
        tf_graph = tf_sess.graph
        with tf_sess.as_default(), tf_graph.as_default():
            self._graph_init_vars(tf_sess)

    ######################
    ### Policy methods ###
    ######################

    def get_action(self, step, current_episode_step, observation, goal, explore, debug=False):
        chosen_actions, chosen_values, action_infos = self.get_actions(step,
                                                                       [current_episode_step],
                                                                       ([observation[0]], [observation[1]]),
                                                                       [goal],
                                                                       explore=explore,
                                                                       debug=debug)
        return chosen_actions[0], chosen_values[0], action_infos[0]

    def get_actions(self, step, current_episode_steps, observations, goals, explore, debug=False):
        assert (not np.isnan(observations[0]).any())
        assert (not np.isnan(observations[1]).any())
        for input_goal_key in self._goals_to_input:
            idx = self._goal_keys.index(input_goal_key)
            assert (not np.isnan(goals)[:, idx].any())

        ds = [{} for _ in range(len(current_episode_steps))]
        observations_im, observations_vec = observations
        feed_dict = {
            self._tf_dict['obs_im_ph']: observations_im,
            self._tf_dict['obs_vec_ph']: observations_vec,
            self._tf_dict['goals_ph']: goals,
            self._tf_dict['episode_timesteps_ph']: current_episode_steps
        }

        if not debug:
            actions, values = self._tf_dict['sess'].run([self._tf_dict['get_action'],
                                                         self._tf_dict['get_action_value']],
                                                        feed_dict=feed_dict)
        else:
            actions, values, actions_all, values_all, yhats_all, bhats_all = \
                self._tf_dict['sess'].run([self._tf_dict['get_action'],
                                           self._tf_dict['get_action_value'],
                                           self._tf_dict['get_action_all'],
                                           self._tf_dict['get_action_value_all'],
                                           self._tf_dict['get_action_yhats_all'],
                                           self._tf_dict['get_action_bhats_all']],
                                          feed_dict=feed_dict)

            for i, d_i in enumerate(ds):
                d_i['actions_all'] = actions_all
                d_i['values_all'] = values_all[i]
                d_i['yhats_all'] = OrderedDict([(k, yhats_all[k][i]) for k in yhats_all.keys()])
                d_i['bhats_all'] = OrderedDict([(k, bhats_all[k][i]) for k in bhats_all.keys()])

        if explore:
            for es in self._exploration_stragies:
                actions = es.add_exploration(step, actions)

        return actions, values, ds
    
    def reset_get_action(self):
        self._tf_dict['sess'].run(self._tf_dict['get_action_reset_ops'])

    #####################
    ### Model methods ###
    #####################

    def get_model_outputs(self, observations, actions, goals):
        observations_im, observations_vec = observations
        feed_dict = {
            self._tf_dict['obs_im_ph']: observations_im,
            self._tf_dict['obs_vec_ph']: observations_vec,
            self._tf_dict['actions_ph']: actions,
            self._tf_dict['goals_ph']: goals,
        }

        yhats, bhats, values, action_value = self._tf_dict['sess'].run([self._tf_dict['yhats'],
                                                                        self._tf_dict['bhats'],
                                                                        self._tf_dict['values'],
                                                                        self._tf_dict['action_value']],
                                                                       feed_dict=feed_dict)

        return yhats, bhats, values, action_value

    ######################
    ### Saving/loading ###
    ######################

    def _saver_ckpt_name(self, ckpt_name, saver_name):
        name, ext = os.path.splitext(ckpt_name)
        saver_ckpt_name = '{0}_{1}{2}'.format(name, saver_name, ext)
        return saver_ckpt_name

    def save(self, ckpt_name, train=True):
        saver_name = 'train' if train else 'inference'
        saver = self._tf_dict['savers_dict'][saver_name]
        saver.save(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name), write_meta_graph=False)

    def restore(self, ckpt_name, train=True, restore_subgraphs=None):
        """
        :param: restore_subgraphs: 'image', 'observation', 'action', 'rnn', 'output
        """
        name = 'train' if train else 'inference'
        if restore_subgraphs is None:
            restore_subgraphs = [name]
        else:
            restore_subgraphs = ['{0}_{1}'.format(name, subgraph) for subgraph in restore_subgraphs]

        for saver_name in restore_subgraphs:
            saver = self._tf_dict['savers_dict'][saver_name]
            saver.restore(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, name))

    def terminate(self):
        self._tf_dict['sess'].close()

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            if k == 'Depth':
                logger.record_tabular(k+'Mean', np.mean(self._log_stats[k]))
                logger.record_tabular(k+'Std', np.std(self._log_stats[k]))
            else:
                logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()
