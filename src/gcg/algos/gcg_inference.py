from gcg.data.timer import timeit
from gcg.data.logger import logger
from gcg.algos.gcg import GCG


class GCGinference(GCG):
    def __init__(self,
                 exp_name,
                 env_params, env_eval_params,
                 rp_params, rp_eval_params,
                 labeller_params,
                 policy_params,
                 alg_params,
                 log_level='info', log_fname='log.txt', seed=None, is_continue=False, params_txt=None):
        env_eval_params = None
        if not alg_params['init_inference_ckpt']:
            print('\n\n!!!!!!!!! No checkpoint being loaded !!!!!!!!!\n\n')
        alg_params['init_train_ckpt'] = None

        super(GCGinference, self).__init__(
            exp_name=exp_name,
            env_params=env_params, env_eval_params=env_eval_params,
            rp_params=rp_params, rp_eval_params=rp_eval_params,
            labeller_params=labeller_params,
            policy_params=policy_params,
            alg_params=alg_params,
            log_level=log_level, log_fname=log_fname, seed=seed, is_continue=is_continue, params_txt=params_txt
        )

    #################
    ### Inference ###
    #################

    def _run_init_inference(self):
        inference_itr = self._fm.get_inference_itr()
        self._restore_rollouts('train')
        self._restore_rollouts('eval')

        save_itr = inference_itr
        start_step = save_itr * self._save_every_n_steps

        timeit.reset()
        timeit.start('total')

        return start_step, save_itr

    def run(self):
        start_step, save_itr = self._run_init_inference()
        last_eval_step = 0

        step = start_step
        while step < self._total_steps:
            step += 1

            if step >= self._sample_after_n_steps:
                step = self._run_env_step(step)

            if step - last_eval_step >= self._eval_every_n_steps and self._replay_pool.finished_storing_rollout:
                self._run_env_eval(step, do_sampler_step=True, calculate_holdout=False)
                last_eval_step = step

            if step % self._log_every_n_steps == 0:
                self._run_log(step)

            if step % self._save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._save_inference(save_itr,
                                     self._replay_pool.get_recent_rollouts(),
                                     self._replay_pool_eval.get_recent_rollouts())
                save_itr += 1

        if step >= self._total_steps:
            logger.info('Saving files for itr {0}'.format(save_itr))
            self._save_inference(save_itr,
                                 self._replay_pool.get_recent_rollouts(),
                                 self._replay_pool_eval.get_recent_rollouts())
