import os
import subprocess

from gcg.samplers.sampler import Sampler
from gcg.data.timer import timeit
from gcg.data.logger import logger
from gcg.data.file_manager import FileManager
from gcg.data import mypickle
from gcg.misc import utils

class GCG(object):

    def __init__(self,
                 exp_name,
                 env_params, env_eval_params,
                 rp_params, rp_eval_params,
                 labeller_params,
                 policy_params,
                 alg_params,
                 log_level='info', log_fname='log.txt', seed=None, is_continue=False, params_txt=None):
        ### create file manager and setup logger
        self._fm = FileManager(exp_name,  is_continue=is_continue, log_level=log_level, log_fname=log_fname)

        logger.debug('Git current')
        logger.debug(subprocess.check_output('git status | head -n 1', shell=True).decode('utf-8').strip())
        logger.debug(subprocess.check_output('git log -n 1| head -n 1', shell=True).decode('utf-8').strip())

        logger.debug('Seed {0}'.format(seed))
        utils.set_seed(seed)

        ### copy params for posterity
        if params_txt:
            with open(self._fm.params_fname, 'w') as f:
                f.write(params_txt)

        ### create environments
        self._env = env_params['class'](params=env_params['kwargs'])
        self._env_eval = env_eval_params['class'](params=env_eval_params['kwargs']) if env_eval_params else self._env

        ### create policy
        self._policy = policy_params['class'](
            env_spec=self._env.spec,
            exploration_strategies=alg_params['exploration_strategies'],
            **policy_params['kwargs']
        )

        ### create labeller
        self._labeller = labeller_params['class'](env_spec=self._env.spec,
                                                  policy=self._policy,
                                                  **labeller_params['kwargs']) if labeller_params['class'] else None

        ### create replay pools
        self._replay_pool = rp_params['class'](env_spec=self._env.spec,
                                               obs_history_len=self._policy.obs_history_len,
                                               N=self._policy.N,
                                               labeller=self._labeller,
                                               **rp_params['kwargs'])
        self._replay_pool_eval = rp_eval_params['class'](env_spec=self._env_eval.spec if self._env_eval else self._env.spec,
                                                         obs_history_len=self._policy.obs_history_len,
                                                         N=self._policy.N,
                                                         labeller=None,
                                                         **rp_eval_params['kwargs']) if rp_eval_params else None

        ### create samplers
        self._sampler = Sampler(
            env=self._env,
            policy=self._policy,
            replay_pool=self._replay_pool
        )
        self._sampler_eval = Sampler(
            env=self._env_eval,
            policy=self._policy,
            replay_pool=self._replay_pool_eval
        ) if self._env_eval is not None and self._replay_pool_eval is not None else None

        ### create algorithm
        self._total_steps = int(alg_params['total_steps'])
        self._sample_after_n_steps = int(alg_params['sample_after_n_steps'])
        self._onpolicy_after_n_steps = int(alg_params['onpolicy_after_n_steps'])
        self._learn_after_n_steps = int(alg_params['learn_after_n_steps'])
        self._train_every_n_steps = alg_params['train_every_n_steps']
        self._eval_every_n_steps = int(alg_params['eval_every_n_steps'])
        self._rollouts_per_eval = int(alg_params.get('rollouts_per_eval', 1))
        self._save_every_n_steps = int(alg_params['save_every_n_steps'])
        self._save_async = alg_params.get('save_async', False)
        self._update_target_after_n_steps = int(alg_params['update_target_after_n_steps'])
        self._update_target_every_n_steps = int(alg_params['update_target_every_n_steps'])
        self._log_every_n_steps = int(alg_params['log_every_n_steps'])
        self._batch_size = alg_params['batch_size']
        if alg_params['offpolicy'] is not None:
            self._add_offpolicy(alg_params['offpolicy'], max_to_add=alg_params['num_offpolicy'])
        if alg_params['init_inference_ckpt'] is not None:
            logger.info('Restoring init_inference_ckpt {0}'.format(alg_params['init_inference_ckpt']))
            self._policy.restore(alg_params['init_inference_ckpt'],
                                 train=False,
                                 restore_subgraphs=alg_params.get('init_inference_restore_subgraphs'))
        if alg_params['init_train_ckpt'] is not None:
            logger.info('Restoring init_train_ckpt {0}'.format(alg_params['init_train_ckpt']))
            self._policy.restore(alg_params['init_train_ckpt'],
                                 train=True,
                                 restore_subgraphs=alg_params.get('init_train_restore_subgraphs'))

    ############
    ### Save ###
    ############

    def _save_train(self, itr):
        self._policy.save(self._fm.train_policy_fname(itr), train=True)
        self._policy.save(self._fm.inference_policy_fname(itr), train=False)

    def _save_inference(self, itr, train_rollouts, eval_rollouts):
        for fname, rollouts in [(self._fm.train_rollouts_fname(itr), train_rollouts),
                                (self._fm.eval_rollouts_fname(itr), eval_rollouts)]:
            assert (not os.path.exists(fname))
            mypickle.dump({'rollouts': rollouts}, fname, async=self._save_async)

    def _save(self, itr, train_rollouts, eval_rollouts):
        self._save_train(itr)
        self._save_inference(itr, train_rollouts, eval_rollouts)

    ###############
    ### Restore ###
    ###############

    def _add_offpolicy(self, folders, max_to_add):
        for folder in folders:
            assert (os.path.exists(folder), 'offpolicy folder {0} does not exist'.format(folder))
            logger.info('Loading offpolicy data from {0}'.format(folder))
            rollout_filenames = [os.path.join(folder, fname) for fname in os.listdir(folder) if 'train_rollouts.pkl' in fname]
            self._replay_pool.store_rollouts(rollout_filenames, max_to_add=max_to_add)
        logger.info('Added {0} samples'.format(len(self._replay_pool)))

    def _restore_rollouts(self, train_or_eval):
        if train_or_eval == 'train':
            rp = self._replay_pool
            fname_func = self._fm.train_rollouts_fname
        elif train_or_eval == 'eval':
            rp = self._replay_pool_eval
            fname_func = self._fm.eval_rollouts_fname
        else:
            raise ValueError('train_or_eval must be train or eval, not {0}'.format(train_or_eval))

        itr = 0
        rollout_filenames = []
        while True:
            fname = fname_func(itr)
            if not os.path.exists(fname):
                break

            rollout_filenames.append(fname)
            itr += 1

        logger.info('Restoring {0} iterations of {1} rollouts....'.format(itr, train_or_eval))
        if rp:
            rp.store_rollouts(rollout_filenames)
        logger.info('Done restoring rollouts!')

    def _restore(self):
        train_itr = self._fm.get_train_itr()
        inference_itr = self._fm.get_inference_itr()
        assert (train_itr == inference_itr,
                'Train itr is {0} but inference itr is {1}'.format(train_itr, inference_itr))

        self._restore_rollouts('train')
        self._restore_rollouts('eval')
        if train_itr > 0:
            self._policy.restore(self._fm.train_policy_fname(train_itr - 1), train=True)

    ########################
    ### Training methods ###
    ########################

    def _run_init(self):
        self._restore()
        # note this is the right step, but the trajectories might not all have been saved
        save_itr = self._fm.get_train_itr()
        start_step = save_itr * self._save_every_n_steps

        self._sampler.reset()

        timeit.reset()
        timeit.start('total')

        return start_step, save_itr

    def _run_env_step(self, step):
        """
        :return: the current step
        """
        timeit.start('sample')
        self._sampler.step(step,
                           take_random_actions=(step < self._onpolicy_after_n_steps),
                           explore=True)
        timeit.stop('sample')

        return step

    def _run_env_eval(self, step, do_sampler_step=True, calculate_holdout=True):
        timeit.start('eval')

        ### add to eval buffer
        if self._sampler_eval and do_sampler_step:
            self._sampler_eval.reset()

            eval_step = step
            num_dones = 0
            while num_dones < self._rollouts_per_eval:
                _, _, _, _, done, _ = \
                    self._sampler_eval.step(eval_step, explore=False)
                eval_step += 1
                num_dones += int(done)

            self._sampler.reset()

        ### calculate holdout costs
        if self._replay_pool and self._replay_pool_eval.can_sample(batch_size=self._batch_size) and calculate_holdout:
            indices, weights, steps, observations, goals, actions, rewards, dones, _ = \
                self._replay_pool_eval.sample(self._batch_size)
            self._policy.eval_holdout(step, steps=steps, observations=observations, goals=goals,
                                      actions=actions, rewards=rewards, dones=dones)

        timeit.stop('eval')

    def _run_train_step(self, step):
        def do_train_step():
            timeit.start('batch')
            indices, weights, steps, observations, goals, actions, rewards, dones, _ = \
                self._replay_pool.sample(self._batch_size)
            timeit.stop('batch')
            timeit.start('train')
            rew_errors = self._policy.train_step(step, steps=steps, observations=observations, goals=goals,
                                                 actions=actions, rewards=rewards, dones=dones, weights=weights)
            self._replay_pool.update_priorities(indices, rew_errors)
            timeit.stop('train')

        if self._train_every_n_steps >= 1:
            if step % int(self._train_every_n_steps) == 0:
                do_train_step()
        else:
            for _ in range(int(1. / self._train_every_n_steps)):
                do_train_step()

        ### update target network
        if step > self._update_target_after_n_steps and step % self._update_target_every_n_steps == 0:
            self._policy.update_target()

    def _run_log(self, step):
        logger.record_tabular('Step', step)
        self._env.log()
        self._replay_pool.log()
        if self._env_eval:
            self._env_eval.log(prefix='Eval')
        if self._replay_pool_eval:
            self._replay_pool_eval.log(prefix='Eval')
        self._policy.log()
        logger.dump_tabular(print_func=logger.info)
        timeit.stop('total')
        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()
        timeit.start('total')

    def _run_save(self, save_itr):
        timeit.start('save')
        logger.info('Saving files for itr {0}'.format(save_itr))
        self._save(save_itr, self._replay_pool.get_recent_rollouts(), self._replay_pool_eval.get_recent_rollouts())
        timeit.stop('save')

    def run(self):

        start_step, save_itr = self._run_init()
        last_eval_step = 0

        step = start_step
        while step < self._total_steps:
            step += 1

            if step >= self._sample_after_n_steps:
                step = self._run_env_step(step)

            if step - last_eval_step >= self._eval_every_n_steps and self._replay_pool.finished_storing_rollout:
                self._run_env_eval(step)
                last_eval_step = step

            if step >= self._learn_after_n_steps and self._replay_pool.can_sample(batch_size=self._batch_size):
                self._run_train_step(step)

            if step % self._save_every_n_steps == 0:
                self._run_save(save_itr)
                save_itr += 1

            if step % self._log_every_n_steps == 0:
                self._run_log(step)

        if step >= self._total_steps:
            self._run_save(save_itr)

