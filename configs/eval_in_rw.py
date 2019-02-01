exp_name = 'ours_rw'
seed = 0
log_level = 'debug'


###################
### Environment ###
###################
from sandbox.crazyflie.src.gcg.envs.crazyflie.crazyflie_env import CrazyflieEnv

env_params = {
    'class': CrazyflieEnv,
    'kwargs': {
        'dt': 0.25,
        'collision_reward_only': True,
        'collision_reward': -1,
        'obs_shape': [72, 96, 1],
        'prompt_save_rollout_on_coll': True
    }
}

env_eval_params = { # if None, env_eval = env
    'class': env_params['class'],
    'kwargs': {
        **env_params['kwargs']
    }
}

###################
### Replay pool ###
###################

from gcg.replay_pools.replay_pool import ReplayPool

rp_params = {
    'class': ReplayPool,
    'kwargs': {
        'size': 1e5,

        'save_rollouts': True, # False saves space
        'save_rollouts_observations': True, # False saves space
        'save_env_infos': True # False saves space
    }
}

rp_eval_params = {
    'class': rp_params['class'],
    'kwargs': {
        **rp_params['kwargs'],
        **{'size': 1e5}
    }
}


################
### Labeller ###
################

labeller_params = {
    'class': None,
    'kwargs': {}
}


#################
### Algorithm ###
#################

from gcg.algos.gcg import GCG

from gcg.exploration_strategies.gaussian_strategy import GaussianStrategy
from gcg.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy

alg_params = {
    'class': GCG,
    'kwargs': {
        ### Offpolicy data ###
        'offpolicy': None,  # folder path containing .pkl files with rollouts
        'num_offpolicy': None,  # number of offpolicy datapoints to load
        'init_train_ckpt': None, # initial training checkpoint model to load from
        'init_inference_ckpt': None, # initial inference checkpoint model to load from

        ### Steps ###

        'total_steps': 8.e+5,  # corresponding to number of env.step(...) calls

        'sample_after_n_steps': -1,
        'onpolicy_after_n_steps': 4.e+3,  # take random actions until this many steps is reached

        'learn_after_n_steps': 1.e+3,  # when to start training the model
        'train_every_n_steps': 0.25, # number of calls to model.train per env.step (if fractional, multiple trains per step)
        'eval_every_n_steps': 5.e+2,  # how often to evaluate policy in env_eval
        'rollouts_per_eval': 1,  # how many rollouts to evaluate per eval_every_n_steps

        'update_target_after_n_steps': -1,  # after which the target network can be updated
        'update_target_every_n_steps': 5.e+3,  # how often to update target network

        'save_every_n_steps': 1.e+4,  # how often to save experiment data
        'log_every_n_steps': 1.e+3,  # how often to print log information

        'batch_size': 32,  # per training step

        ### Exploration ###

        'exploration_strategies': [
            {
                'class': GaussianStrategy,
                'params': {
                    # endpoints: [[step, value], [step, value], ...]
                    'endpoints': [[0, 0.25], [8.e+4, 0.05], [24.e+4, 0.005]],
                    'outside_value': 0.005
                },
            },
            {
                'class': EpsilonGreedyStrategy,
                'params': {
                    # endpoints: [[step, value], [step, value], ...]
                    'endpoints': [[0, 1.0], [1.e+3, 1.0], [8.e+4, 0.1], [16.e+4, 0.01]],
                    'outside_value': 0.01
                },
            },
        ],
    }
}

##############
### Policy ###
##############

from gcg.policies.gcg_policy_tfrecord import GCGPolicyTfrecord
import tensorflow as tf
from gcg.tf.layers.cnn.convolution import Convolution
from gcg.tf.layers.fullyconnectednn.fully_connected import FullyConnected
from gcg.tf.layers.rnn.rnn_cell import DpMulintLSTMCell

policy_params = {
    'class': GCGPolicyTfrecord,  # <GCGPolicy> model class

    'kwargs': {
        'N': 12,  # label horizon
        'H': 12,  # model horizon
        'gamma': 0.95,  # discount factor
        'obs_history_len': 4,  # number of previous observations to concatenate (inclusive)
        'inference_only': True,
        'use_target': False,  # target network?
        'goals_to_input': [],

        'goal_in_obs': False,  # is the goal included as input? (concatenated to obs_vec)
        # actions, yhats, bhats, values, goals are available
        # action_selection_value
        # :param actions
        # :param yhats
        # :param bhats
        # :param values
        # :param goals
        'action_selection_value': lambda actions, yhats, bhats, values, goals: \
            -values['coll'] + \
             0.01/ 180. * tf.reduce_mean(-tf.square(actions['yaw']), axis=1, keep_dims=True),
            # -0.01 / 180.0 * tf.expand_dims(tf.reduce_mean(tf.abs(actions['yaw']), axis=1), 1), # 0.05

        'outputs': [
            {
                'name': 'coll',
                # yhat
                # :param pre_yhat: [batch_size, H]
                # :param obs_vec: {name : [batch_size, 1]}
                'yhat': lambda pre_yhat, obs_vec: tf.nn.sigmoid(pre_yhat),

                # yhat_label
                # :param rewards: [batch_size, N]
                # :param dones: [batch_size, N+1]
                # :param goals: {name: [batch_size, 1]}
                # :param target_obs_vec: [batch_size, N]
                # :param gamma: scalar
                'yhat_label': lambda rewards, dones, goals, future_goals, target_obs_vec, gamma: \
                    tf.cast(tf.cumsum(target_obs_vec['coll'], axis=1) >= 1.0, tf.float32),

                # yhat training cost
                'yhat_loss': 'xentropy',  # <mse / huber / xentropy>
                'yhat_loss_weight': 1.0,  # how much to weight this loss compared to other losses
                'yhat_loss_use_pre': True,  # use the pre-activation for the loss? needed for xentropy
                'yhat_loss_xentropy_posweight': 10.0,  # larger value --> false negatives cost more

                # bhat
                # :param pre_bhat: [batch_size, H]
                # :param obs_vec: {name: [batch_size, 1]}
                'bhat': None,

                # bhat_label
                # :param rewards: [batch_size, N]
                # :param dones: [batch_size, N+1]
                # :param goals: {name: [batch_size, 1]}
                # :param target_obs_vec: {name: [batch_size, H]}
                # :param gamma: scalar
                # :param future_goals: {name: [batch_size, N]}
                # :param target_yhats: {name: [batch_size, N, H]}
                # :param target_bhats: {name: [batch_size, N, H]}
                # :param target_values: {name: [batch_size, N]}
                'bhat_label': None,

                # bhat training cost
                'bhat_loss': None,  # <mse / huber / xentropy>
                'bhat_loss_weight': None,  # how much to weight this loss compared to other losses
                'bhat_loss_use_pre': None,  # use the pre-activation for the loss? needed for xentropy
                'bhat_loss_xentropy_posweight': None,  # larger value --> false negatives cost more


                # value
                # :param yhats: {name: [batch_size, H]}
                # :param bhats: {name: [batch_size, H]}
                # :param goals: {name: [batch_size, 1]}
                # :param gamma
                'value': lambda yhats, bhats, goals, gamma: tf.reduce_mean(yhats['coll'], axis=1),

                # do you train RNN beyond dones?
                'clip_with_done': False,
            }
        ],

        ### Action selection

        'get_action_test': {  # how to select actions at test time (i.e., when gathering samples)
            'H': 12,
            'h_select': 0,
            'type': 'cem',  # <random/cem> action selection method
            'random': {
                'K': 4096,
            },
            'cem': {
                'M_init': 4096,
                'M': 1024,
                'K': 128,
                'itrs': 3,
                'eps': 1.e-2,
            }
        },

        'get_action_target': {
            'H': 12,
            'type': 'random',  # <random>

            'random': {
                'K': 100,
            },
        },

        ### Network architecture

        'image_graph': {  # CNN
            'conv_class': Convolution,
            'conv_args': {},
            'filters': [64, 32, 32, 32],
            'kernels': [8, 4, 3, 3],
            'strides': [4, 2, 2, 2],
            'padding': 'SAME',
            'hidden_activation': tf.nn.relu,
            'output_activation': tf.nn.relu,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': False
        },

        'observation_graph': {  # fully connected
            'fullyconnected_class': FullyConnected,
            'fullyconnected_args': {},
            'hidden_layers': [256],
            'output_dim': 128,  # this is the hidden size of the rnn
            'hidden_activation': tf.nn.relu,
            'output_activation': tf.nn.relu,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': False,
        },

        'action_graph': {  # fully connected
            'fullyconnected_class': FullyConnected,
            'fullyconnected_args': {},
            'hidden_layers': [16],
            'output_dim': 16,
            'hidden_activation': tf.nn.relu,
            'output_activation': tf.nn.relu,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': True,
        },

        'rnn_graph': {
            'rnncell_class': DpMulintLSTMCell,
            'rnncell_args': {},
            'num_cells': 1,
            'state_tuple_size': 2,  # 1 for standard cells, 2 for LSTM
            'trainable': True
        },

        'output_graph': {  # fully connected
            'fullyconnected_class': FullyConnected,
            'fullyconnected_args': {},
            'hidden_layers': [16],
            # 'output_dim': None, # is determined by yhat / bhat
            'hidden_activation': tf.nn.relu,
            'output_activation': None,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': True,
        },

        ### Training

        'optimizer': 'adam',  # <adam/sgd>
        'weight_decay': 1e-5,  # L2 regularization
        'lr_schedule': {  # learning rate schedule
            'endpoints': [],
            'outside_value': 1.e-4,
        },
        'grad_clip_norm': 10,  # clip the gradient magnitude

        ### Device
        'gpu_device': 0,
        'gpu_frac': 0.5,
        'seed': 0
    },
}

##################
### Evaluation ###
##################

from sandbox.crazyflie.src.gcg.algos.eval_crazyflie_gcg import EvalCrazyflieGCG

eval_params = {
    'class': EvalCrazyflieGCG,
    'kwargs': {
    }
}


params = {
    'exp_name': exp_name,
    'seed': seed,
    'log_level': log_level,

    'env': env_params,
    'env_eval': env_eval_params,

    'replay_pool': rp_params,
    'replay_pool_eval': rp_eval_params,

    'labeller': labeller_params,

    'alg': alg_params,

    'policy': policy_params,

    'eval': eval_params
}
