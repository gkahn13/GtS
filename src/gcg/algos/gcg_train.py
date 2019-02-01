from gcg.data.timer import timeit
from gcg.data.logger import logger
from gcg.algos.gcg import GCG


class GCGtrain(GCG):
    def __init__(self,
                 exp_name,
                 env_params, env_eval_params,
                 rp_params, rp_eval_params,
                 labeller_params,
                 policy_params,
                 alg_params,
                 log_level='info', log_fname='log.txt', seed=None, is_continue=False, params_txt=None):
        assert len(alg_params['offpolicy']) > 0, 'Must pass in training data for GCGtrain'

        super(GCGtrain, self).__init__(
            exp_name=exp_name,
            env_params=env_params, env_eval_params=env_eval_params,
            rp_params=rp_params, rp_eval_params=rp_eval_params,
            labeller_params=labeller_params,
            policy_params=policy_params,
            alg_params=alg_params,
            log_level=log_level, log_fname=log_fname, seed=seed, is_continue=is_continue, params_txt=params_txt
        )

    #############
    ### Train ###
    #############

    def _run_init_train(self):
        train_itr = self._fm.get_train_itr()
        if train_itr > 0:
            logger.info('Restore train iteration {0}'.format(train_itr - 1))
            self._policy.restore(self._fm.train_policy_fname(train_itr - 1), train=True)

        save_itr = train_itr
        start_step = save_itr * self._save_every_n_steps

        timeit.reset()
        timeit.start('total')

        return start_step, save_itr

    def run(self):
        start_step, save_itr = self._run_init_train()

        step = start_step
        while step < self._total_steps:
            step += 1

            if step % self._eval_every_n_steps == 0:
                self._run_env_eval(step, do_sampler_step=False, calculate_holdout=True)

            if step >= self._learn_after_n_steps:
                self._run_train_step(step)

            if step % self._log_every_n_steps == 0:
                self._run_log(step)

            if step % self._save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._save_train(save_itr)
                save_itr += 1

        if step >= self._total_steps:
            logger.info('Saving files for itr {0}'.format(save_itr))
            self._save_train(save_itr)
