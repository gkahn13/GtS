import os, copy, glob
import yaml
import itertools

import pandas
import numpy as np
import matplotlib.pyplot as plt

from gcg.data import mypickle
from . import utils
from gcg.data.file_manager import FileManager
from gcg.misc.utils import import_params

############################
### Multiple Experiments ###
############################

class MultiExperimentComparison(object):
    def __init__(self, experiment_groups):
        self._experiment_groups = experiment_groups

    def __getitem__(self, item):
        """
        return experiment with params['exp_name'] == item
        """
        for exp in self.list:
            if exp.name == item:
                return exp

        raise Exception('Experiment {0} not found'.format(item))

    @property
    def list(self):
        return list(itertools.chain(*[eg.experiments for eg in self._experiment_groups]))

    ################
    ### Plotting ###
    ################

    def plot_csv(self, keys, save_path=None, xkey='Step', **kwargs):
        """
        :param keys: which keys from the csvs do you want to plot
        :param kwargs: save_path, plot_std, avg_window, xlim, ylim
        """
        num_plots = len(keys)

        f, axes = plt.subplots(1, num_plots, figsize=5*np.array([num_plots, 1]))
        if not hasattr(axes, '__iter__'):
            axes = [axes]

        for ax, key in zip(axes[:num_plots], keys):
            self._plot_csv(ax, key, xkey=xkey, **kwargs)

        if save_path is None:
            plt.show(block=True)
        else:
            f.savefig(save_path,
                      bbox_inches='tight',
                      dpi=kwargs.get('dpi', 100))

    def _plot_csv(self, ax, key, xkey='Step', **kwargs):
        avg_window = kwargs.get('avg_window', None)
        plot_std = kwargs.get('plot_std', True)
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)

        for experiment_group in self._experiment_groups:
            csvs = experiment_group.csv

            data_interp = utils.DataAverageInterpolation()
            data_interp_stds = utils.DataAverageInterpolation()
            min_step, max_step = -np.inf, np.inf
            for csv in csvs:
                steps, values = csv[xkey], csv[key]
                if avg_window is not None:
                    steps, values, values_stds = utils.moving_avg_std(steps, values, window=avg_window)
                    data_interp_stds.add_data(steps, values_stds)
                data_interp.add_data(steps, values)

                min_step = max(min_step, min(steps))
                max_step = min(max_step, max(steps))

            steps = np.linspace(min_step, max_step, int(1e3))[1:-1]
            values_mean, _ = data_interp.eval(steps)
            if avg_window is not None:
                values_std, _ = data_interp_stds.eval(steps)

            ax.plot(steps, values_mean, **experiment_group.plot)
            if plot_std:
                plot_params = copy.deepcopy(experiment_group.plot)
                plot_params['label'] = None
                ax.fill_between(steps, values_mean - values_std, values_mean + values_std, alpha=0.4, **plot_params)
            ax.set_xlabel(xkey)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_title(key, fontdict={'fontsize': 8})

        ax.legend()



########################################
### Single Experiment Multiple Seeds ###
########################################

class ExperimentGroup(object):
    def __init__(self, exp_names, plot=dict(), clear_obs=False, label_params=None):
        self.plot = plot

        ### load experiments
        self.experiments = [Experiment(exp_name, plot=plot, clear_obs=clear_obs) for exp_name in exp_names]

        if label_params is not None:
            label = self.get_plot_label(label_params)
            for plot in [self.plot] + [exp.plot for exp in self.experiments]:
                plot['label'] = label


    ####################
    ### Data loading ###
    ####################

    @property
    def params(self):
        return [exp.params for exp in self.experiments]

    @property
    def name(self):
        return [exp.name for exp in self.experiments]

    @property
    def csv(self):
        return [exp.csv for exp in self.experiments]

    @property
    def train_rollouts(self):
        return [exp.train_rollouts for exp in self.experiments]

    @property
    def eval_rollouts(self):
        return [exp.eval_rollouts for exp in self.experiments]

    def get_plot_label(self, label_params):
            """
            :param keys_list: list of keys from self.params to generate label from
            e.g.
                [('policy', 'H'), ('policy', 'get_action', 'K'), ...]
            :return: str
            """
            def nested_get(dct, keys):
                for key in keys:
                    dct = dct[key]
                return dct

            label = ', '.join(['{0}: {1}'.format(k, nested_get(self.params[0], v)) for k, v in label_params])
            return label


#########################
### Single Experiment ###
#########################

class Experiment(object):
    def __init__(self, exp_name, plot=dict(), clear_obs=False):
        self._exp_name = exp_name
        self._fm = FileManager(exp_name, is_continue=True, log_level='debug', log_fname='/tmp/log_post_experiment.txt')

        self.plot = plot
        self._clear_obs = clear_obs

        self._internal_params = None
        self._internal_csv = None
        self._internal_train_rollouts = None
        self._internal_eval_rollouts = None

        self.env = None
        self.policy = None

    ########################
    ### Env and policies ###
    ########################

    def create_env(self):
        if self.env is None:
            env_params = self.params['env']
            self.env = env_params['class'](params=env_params['kwargs'])

    def create_policy(self, gpu_device=None, gpu_frac=None):
        assert (self.env is not None)
        policy_params = self.params['policy']

        if gpu_device is not None:
            policy_params['gpu_device'] = gpu_device
        if gpu_frac is not None:
            policy_params['gpu_frac'] = gpu_frac

        self.policy = policy_params['class'](
            env_spec=self.env.spec,
            exploration_strategies=[],
            inference_only=True,
            **policy_params['kwargs'],
        )

    def restore_policy(self, itr=None):
        assert (self.policy is not None)

        if itr is None:
            itr = 0
            while len(glob.glob(self._fm.inference_policy_fname(itr) + '*')) > 0:
                itr += 1
            itr -= 1

        if itr >= 0:
            print('Loading train policy from iteration {0}...'.format(itr))
            self.policy.restore(self._fm.inference_policy_fname(itr), train=False)

        return itr

    def close_policy(self):
        self.policy.terminate()
        self.policy = None

    #############
    ### Files ###
    #############

    @property
    def _csv_file(self):
        log = os.path.join(self._fm.exp_dir, 'log.csv')
        log_train = os.path.join(self._fm.exp_dir, 'log_train.csv')
        log_inference = os.path.join(self._fm.exp_dir, 'log_inference.csv')

        exists_log = os.path.exists(log)
        exists_log_train = os.path.exists(log_train)
        exists_log_inference = os.path.exists(log_inference)

        assert (sum([exists_log, exists_log_train, exists_log_inference]) == 1)

        if exists_log:
            return log
        if exists_log_train:
            return log_train
        if exists_log_inference:
            return log_inference

    ####################
    ### Data loading ###
    ####################

    @property
    def params(self):
        if self._internal_params is None:
            self._internal_params = import_params(self._fm.params_fname)

        return self._internal_params

    @property
    def name(self):
        return self._exp_name

    @property
    def csv(self):
        if self._internal_csv is None:
            self._internal_csv = pandas.read_csv(self._csv_file)

            save_every_n_steps = self.params['alg']['kwargs']['save_every_n_steps']
            itrs = (self._internal_csv['Step'] - save_every_n_steps) / save_every_n_steps
            self._internal_csv['itr'] = itrs

        return self._internal_csv

    @property
    def train_rollouts(self):
        if self._internal_train_rollouts is None:
            self._internal_train_rollouts = self._load_rollouts(self._fm.train_rollouts_fname)

        return self._internal_train_rollouts

    @property
    def eval_rollouts(self):
        if self._internal_eval_rollouts is None:
            self._internal_eval_rollouts = self._load_rollouts(self._fm.eval_rollouts_fname)

        return self._internal_eval_rollouts

    def _load_rollouts(self, file_func):
        rollouts_itrs = []
        itr = 0
        while os.path.exists(file_func(itr)):
            rollouts = mypickle.load(file_func(itr))['rollouts']
            if self._clear_obs:
                for r in rollouts:
                    r['observations'] = None
            rollouts_itrs.append(rollouts)
            itr += 1

        return rollouts_itrs