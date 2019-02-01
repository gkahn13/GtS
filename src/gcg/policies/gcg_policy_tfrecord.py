import os, random
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from gcg.policies.gcg_policy import GCGPolicy
from gcg.data.file_manager import FileManager
from gcg.data.logger import logger
from gcg.tf import tf_utils


class GCGPolicyTfrecord(GCGPolicy):
    tfrecord_feature_names = ('observations_im', 'observations_vec', 'actions', 'dones', 'goals', 'rewards')
    tfrecord_feature_tf_types = (tf.uint8, tf.float32, tf.float32, tf.uint8, tf.float32, tf.float32)
    tfrecord_feature_np_types = (np.uint8, np.float32, np.float32, np.uint8, np.float32, np.float32)

    def __init__(self, **kwargs):
        self._batch_size = None
        self._tfrecord_train_fnames = []
        self._tfrecord_holdout_fnames = []

        inference_only = kwargs.get('inference_only', False)
        if not inference_only:
            self._batch_size = kwargs['batch_size']

            for folder in kwargs['tfrecord_folders']:
                tfrecord_fnames = [os.path.join(folder, fname) for fname in os.listdir(folder)
                                    if os.path.splitext(fname)[1] == '.tfrecord']

                for tfrecord_fname in tfrecord_fnames:
                    logger.debug('Tfrecord {0}'.format(tfrecord_fname))
                    if os.path.splitext(FileManager.train_rollouts_fname_suffix)[0] in os.path.splitext(tfrecord_fname)[0]:
                        self._tfrecord_train_fnames.append(tfrecord_fname)
                    elif os.path.splitext(FileManager.eval_rollouts_fname_suffix)[0] in os.path.splitext(tfrecord_fname)[0]:
                        self._tfrecord_holdout_fnames.append(tfrecord_fname)
                    else:
                        raise ValueError('tfrecord {0} does not end in {1} or {2}'.format(
                            tfrecord_fname,
                            os.path.splitext(FileManager.train_rollouts_fname_suffix)[0],
                            os.path.splitext(FileManager.eval_rollouts_fname_suffix)[0])
                        )

            random.shuffle(self._tfrecord_train_fnames)
            random.shuffle(self._tfrecord_holdout_fnames)

        super(GCGPolicyTfrecord, self).__init__(**kwargs)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_input_output_placeholders(self):
        with tf.variable_scope('input_output_placeholders'):
            ### policy inputs
            tf_obs_im_ph = tf.placeholder(tf.uint8, [None, self._obs_history_len, self._obs_im_dim], name='tf_obs_im_ph')
            tf_obs_vec_ph = tf.placeholder(tf.float32, [None, self._obs_history_len, self._obs_vec_dim], name='tf_obs_vec_ph')
            tf_actions_ph = tf.placeholder(tf.float32, [None, self._H, self._action_dim], name='tf_actions_ph')
            tf_goals_ph = tf.placeholder(tf.float32, [None, self._goal_dim], name='tf_goals_ph')
            ### episode timesteps
            tf_episode_timesteps_ph = tf.placeholder(tf.int32, [None], name='tf_episode_timesteps')

        return tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_goals_ph, tf_episode_timesteps_ph

    def _graph_input_output_tfrecord(self, tfrecord_fnames,
                                     num_parallel_calls=4, shuffle_buffer_size=10000, prefetch_buffer_size_multiplier=4):
        def parse(ex):
            sequence_features = {k: tf.FixedLenSequenceFeature([1], dtype=tf.string)
                                 for k in GCGPolicyTfrecord.tfrecord_feature_names}

            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=ex,
                sequence_features=sequence_features
            )

            sequence_parsed = {k: tf.decode_raw(sequence_parsed[k], tf_dtype)[:, 0] for k, tf_dtype in
                               zip(GCGPolicyTfrecord.tfrecord_feature_names, GCGPolicyTfrecord.tfrecord_feature_tf_types)}

            def set_shape(tensor, shape):
                if len(shape) == 0:
                    tensor.set_shape([None, 1])
                    tensor = tensor[:, 0]
                else:
                    tensor.set_shape([None] + shape)
                return tensor

            for k, shape in [('observations_im', [self._obs_im_dim]),
                             ('observations_vec', [self._obs_vec_dim]),
                             ('actions', [self._action_dim]),
                             ('dones', []),
                             ('goals', [self._goal_dim]),
                             ('rewards', [])]:
                sequence_parsed[k] = set_shape(sequence_parsed[k], shape)

            return [sequence_parsed[k] for k in GCGPolicyTfrecord.tfrecord_feature_names]

        def pad(observations_im, observations_vec, actions, dones, goals, rewards):
            d = {
                'observations_im': observations_im,
                'observations_vec': observations_vec,
                'actions': actions,
                'dones': dones,
                'goals': goals,
                'rewards': rewards
            }

            # pre-pend obs_history_len - 1
            for k, v in d.items():
                if len(v.get_shape()) == 1:
                    pad_shape = [self._obs_history_len]
                elif len(v.get_shape()) == 2:
                    pad_shape = [self._obs_history_len, v.get_shape()[1].value]
                else:
                    raise ValueError
                d[k] = tf.concat([tf.zeros(pad_shape, dtype=v.dtype), v], axis=0)

            # post-pend N

            v = d['observations_im']
            d['observations_im'] = tf.concat([v, tf.zeros([self._N, v.get_shape()[1].value], dtype=v.dtype)], axis=0)

            v = d['observations_vec']
            d['observations_vec'] = tf.concat([v, tf.zeros([self._N, v.get_shape()[1].value], dtype=v.dtype)], axis=0)

            v = d['rewards']
            d['rewards'] = tf.concat([v, tf.zeros(self._N, dtype=v.dtype)], axis=0)

            v = d['dones']
            d['dones'] = tf.concat([v, tf.ones(self._N, dtype=v.dtype)], axis=0)

            v = d['actions']
            d['actions'] = tf.concat([v, self._graph_generate_random_actions([self._N])], axis=0)

            v = d['goals']
            d['goals'] = tf.concat([v, tf.tile(tf.expand_dims(v[-1], axis=0), (self._N, 1)),], axis=0)


            return [d[k] for k in GCGPolicyTfrecord.tfrecord_feature_names]

        subsequence_length = self._obs_history_len + self._N
        def subsequence_generator(observations_im, observations_vec, actions, dones, goals, rewards):
            d = {
                'observations_im': observations_im,
                'observations_vec': observations_vec,
                'actions': actions,
                'dones': dones,
                'goals': goals,
                'rewards': rewards
            }

            for k, v in d.items():
                d[k] = np.array([v[h-self._obs_history_len+1:h+self._N+1]
                                 for h in range(self._obs_history_len, len(v) - self._N - 1)])
                assert d[k].shape[1] == subsequence_length

            return [d[k] for k in GCGPolicyTfrecord.tfrecord_feature_names]

        dataset = tf.data.TFRecordDataset(tfrecord_fnames)
        dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls)
        dataset_shapes = dataset.output_shapes
        dataset = dataset.map(pad, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(lambda observations_im, observations_vec, actions, dones, goals, rewards:
                              tf.py_func(subsequence_generator,
                                         [observations_im, observations_vec, actions, dones, goals, rewards],
                                         Tout=GCGPolicyTfrecord.tfrecord_feature_tf_types),
                              num_parallel_calls=num_parallel_calls)
        dataset_shapes = [tf.TensorShape(s.as_list()[:1] + [subsequence_length] + s.as_list()[1:]) for s in dataset_shapes]
        dataset = dataset.apply(tf.contrib.data.unbatch())
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer_size))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size_multiplier * self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()
        for next_element_i, shape_i in zip(next_element, dataset_shapes):
            next_element_i.set_shape(shape_i)

        dataset_init_op = iterator.make_initializer(dataset)

        assert GCGPolicyTfrecord.tfrecord_feature_names == \
               ('observations_im', 'observations_vec', 'actions', 'dones', 'goals', 'rewards')

        ss_obs_im, ss_obs_vec, ss_actions, ss_dones, ss_goals, ss_rewards = next_element

        ss_dones = tf.cast(ss_dones, tf.bool)

        tf_obs_im_tr = ss_obs_im[:, :self._obs_history_len, :]
        tf_obs_vec_tr = ss_obs_vec[:, :self._obs_history_len, :]
        tf_actions_tr = ss_actions[:, self._obs_history_len-1:self._obs_history_len-1+self._H, :]
        tf_dones_tr = ss_dones[:, self._obs_history_len-1:self._obs_history_len-1+self._N+1]
        tf_goals_tr = ss_goals[:, self._obs_history_len-1, :]
        tf_rewards_tr = ss_rewards[:, self._obs_history_len-1:self._obs_history_len-1+self._N]
        tf_weights_tr = tf.ones(self._batch_size, dtype=tf.float32)
        tf_obs_im_target_tr = ss_obs_im
        tf_obs_vec_target_tr = ss_obs_vec
        tf_goals_target_tr = ss_goals[:, self._obs_history_len-1:self._obs_history_len-1+self._N+1]
        tf_episode_timesteps_tr = None # fill this in if we ever do use it

        return dataset_init_op, tf_obs_im_tr, tf_obs_vec_tr, tf_actions_tr, tf_dones_tr, tf_goals_tr, tf_rewards_tr, \
               tf_weights_tr, tf_obs_im_target_tr, tf_obs_vec_target_tr, tf_goals_target_tr, tf_episode_timesteps_tr

    ### high-level
    
    def _graph_setup_feed_dict(self):
        """
        this part of the setup is needed for get_actions(...) and get_model_outputs(...)
        """
        
        ### create input output placeholders
        tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_goals_ph, tf_episode_timesteps_ph = \
            self._graph_input_output_placeholders()
        self.global_step = tf.Variable(0, trainable=False, name='{0}/global_step'.format(self._gcg_scope))

        obs_vec = OrderedDict()
        for i, key in enumerate(self._obs_vec_keys):
            obs_vec[key] = tf_obs_vec_ph[:, 0, i:i + 1]

        goals = OrderedDict()
        for i, key in enumerate(self._goal_keys):
            goals[key] = tf_goals_ph[:, i:i + 1]

        actions = OrderedDict()
        for i, key in enumerate(self._action_keys):
            actions[key] = tf_actions_ph[:, :, i]

        ### setup policy
        values, yhats, bhats, pre_yhats, pre_bhats = \
            self._graph_setup_policy(obs_vec, goals, tf_obs_im_ph, tf_obs_vec_ph, tf_goals_ph, tf_actions_ph,
                                     reuse=False, is_training=False)
        action_value = self._get_action_value(tf_actions_ph, values, yhats, bhats, obs_vec, goals)

        ### get action
        tf_get_action, tf_get_action_value, \
        tf_get_action_reset_ops, \
        tf_get_action_all, tf_get_action_value_all, tf_get_action_yhats_all, tf_get_action_bhats_all = \
            self._graph_setup_action_selection(tf_obs_im_ph, tf_obs_vec_ph,
                                               tf_goals_ph, obs_vec, goals, tf_episode_timesteps_ph)
        
        return {
            'obs_im_ph': tf_obs_im_ph,
            'obs_vec_ph': tf_obs_vec_ph,
            'goals_ph': tf_goals_ph,
            'actions_ph': tf_actions_ph,
            'episode_timesteps_ph': tf_episode_timesteps_ph,
            'get_action': tf_get_action,
            'get_action_value': tf_get_action_value,
            'get_action_reset_ops': tf_get_action_reset_ops,
            'get_action_all': tf_get_action_all,
            'get_action_value_all': tf_get_action_value_all,
            'get_action_yhats_all': tf_get_action_yhats_all,
            'get_action_bhats_all': tf_get_action_bhats_all,
            'yhats': yhats,
            'bhats': bhats,
            'values': values,
            'action_value': action_value,
        }

    def _graph_setup_tfrecord(self, is_train):
        tfrecord_fnames = self._tfrecord_train_fnames if is_train else self._tfrecord_holdout_fnames
        dataset_init_op, tf_obs_im_tr, tf_obs_vec_tr, tf_actions_tr, tf_dones_tr, tf_goals_tr, tf_rewards_tr, \
        tf_weights_tr, tf_obs_im_target_tr, tf_obs_vec_target_tr, tf_goals_target_tr, tf_episode_timesteps_tr = \
            self._graph_input_output_tfrecord(tfrecord_fnames)

        obs_vec, goals, future_goals, actions = self._graph_setup_ordered_dicts(tf_obs_vec_tr,
                                                                                tf_goals_tr,
                                                                                tf_goals_target_tr,
                                                                                tf_actions_tr)

        ### setup policy
        policy_scope = 'policy'
        values, yhats, bhats, pre_yhats, pre_bhats = \
            self._graph_setup_policy(obs_vec, goals, tf_obs_im_tr, tf_obs_vec_tr, tf_goals_tr, tf_actions_tr,
                                     reuse=True)

        ### get policy variables
        tf_inference_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                     scope='{0}/{1}'.format(self._gcg_scope, policy_scope)),
                                   key=lambda v: v.name)
        tf_trainable_policy_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='{0}/{1}'.format(self._gcg_scope, policy_scope)),
                                          key=lambda v: v.name)
        assert (len(tf_trainable_policy_vars) > 0)

        if not self._inference_only:
            target_obs_vec = OrderedDict()
            for i, key in enumerate(self._obs_vec_keys):
                target_obs_vec[key] = tf_obs_vec_target_tr[:, -self._N:, i]

            ### setup target
            target_values, target_yhats, target_bhats, tf_target_vars, tf_update_target_fn = \
                self._graph_setup_target(tf_obs_im_target_tr, tf_obs_vec_target_tr, tf_goals_target_tr, target_obs_vec,
                                         goals, tf_inference_vars, reuse_eval=not is_train)

            ### optimization
            tf_cost_with_reg, tf_cost, tf_costs, tf_costs_yhat, tf_costs_bhat, tf_accs_yhat, tf_rew_errors = \
                self._graph_cost(values, yhats, bhats, pre_yhats, pre_bhats, obs_vec, goals, future_goals,
                                 actions, tf_obs_vec_target_tr, tf_rewards_tr, tf_dones_tr,
                                 target_obs_vec, target_values, target_yhats, target_bhats, tf_weights_tr)

            if is_train:
                tf_opt, tf_lr_ph = self._graph_optimize(tf_cost_with_reg, tf_trainable_policy_vars)
            else:
                tf_opt = tf_lr_ph = None
        else:
            tf_cost_with_reg, tf_cost, tf_costs, tf_costs_yhat, tf_costs_bhat, tf_accs_yhat = \
                None, None, dict(), dict(), dict(), dict()
            tf_target_vars = tf_update_target_fn = tf_cost = tf_opt = tf_lr_ph = tf_rew_errors = None

        if is_train:
            return {
                'dataset_init_op': dataset_init_op,
                'cost_with_reg': tf_cost_with_reg,
                'cost': tf_cost,
                'costs': tf_costs,
                'costs_yhat': tf_costs_yhat,
                'costs_bhat': tf_costs_bhat,
                'accs_yhat': tf_accs_yhat,
                'rew_errors': tf_rew_errors,
                'lr_ph': tf_lr_ph,
                'opt': tf_opt,
                'update_target_fn': tf_update_target_fn
            }
        else:
            return {
                'holdout_dataset_init_op': dataset_init_op,
                'holdout_cost_with_reg': tf_cost_with_reg,
                'holdout_cost': tf_cost,
                'holdout_costs': tf_costs,
                'holdout_costs_yhat': tf_costs_yhat,
                'holdout_costs_bhat': tf_costs_bhat,
                'holdout_accs_yhat': tf_accs_yhat,
                'holdout_rew_errors': tf_rew_errors,
            }

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = tf_utils.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            tf.set_random_seed(self._seed)
            with tf.variable_scope(self._gcg_scope):
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                d_feed_dict = self._graph_setup_feed_dict()

                ### get policy variables
                tf_inference_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                             scope='{0}/{1}'.format(self._gcg_scope, self._policy_scope)),
                                           key=lambda v: v.name)

                if not self._inference_only:
                    d_tfrecord_train = self._graph_setup_tfrecord(is_train=True)
                    d_tfrecord_holdout = self._graph_setup_tfrecord(is_train=False)
                else:
                    d_tfrecord_train = dict()
                    d_tfrecord_holdout = dict()

            ### savers
            tf_all_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._gcg_scope),
                                 key=lambda v: v.name)
            tf_savers_dict = self._graph_setup_savers(tf_inference_vars, tf_all_vars, self._inference_only)

            ### initialize
            self._graph_init_vars(tf_sess)
            if not self._inference_only:
                tf_sess.run([d_tfrecord_train['dataset_init_op'], d_tfrecord_holdout['holdout_dataset_init_op']])

        assert len(set(list(d_feed_dict.keys()) + list(d_tfrecord_train.keys()) + list(d_tfrecord_holdout.keys()))) == \
               len(d_feed_dict.keys()) + len(d_tfrecord_train.keys()) + len(d_tfrecord_holdout.keys()), \
               'Some overlap in the keys...'
        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'savers_dict': tf_savers_dict,
            **d_feed_dict,
            **d_tfrecord_train,
            **d_tfrecord_holdout
        }


    ################
    ### Training ###
    ################

    def train_step(self, step):
        feed_dict = {
            self._tf_dict['lr_ph']: self._lr_schedule.value(step),
        }

        # print('step: {0}'.format(step))
        # debug_keys = list(self._tf_debug.keys())
        # debug_outputs = self._tf_dict['sess'].run([self._tf_debug[k] for k in debug_keys])
        # d = {k: v for k, v in zip(debug_keys, debug_outputs)}
        #
        # import matplotlib.pyplot as plt
        # for vae_in, vae_out in zip(d['vae_input'], d['vae_reconstr_mean']):
        #     vae_out = np.reshape(vae_out, vae_in.shape)
        #     f, axes = plt.subplots(2, self._obs_history_len)
        #     for i, ax in enumerate(axes[0]):
        #         ax.imshow(vae_in[..., i], cmap='Greys_r')
        #     for i, ax in enumerate(axes[1]):
        #         ax.imshow(vae_out[..., i], cmap='Greys_r')
        #     plt.show()
        #
        # import IPython; IPython.embed()


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

    def eval_holdout(self):
        cost_with_reg, cost, costs, costs_yhat, costs_bhat, accs_yhat = \
            self._tf_dict['sess'].run([self._tf_dict['holdout_cost_with_reg'],
                                       self._tf_dict['holdout_cost'],
                                       self._tf_dict['holdout_costs'],
                                       self._tf_dict['holdout_costs_yhat'],
                                       self._tf_dict['holdout_costs_bhat'],
                                       self._tf_dict['holdout_accs_yhat']])

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
