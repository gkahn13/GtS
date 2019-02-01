from gcg.algos.gcg_train import GCGtrain
from gcg.data.timer import timeit

from gcg.policies.gcg_policy_tfrecord import GCGPolicyTfrecord

class GCGtrainTfrecord(GCGtrain):
    def __init__(self,
                 exp_name,
                 env_params, env_eval_params,
                 rp_params, rp_eval_params,
                 labeller_params,
                 policy_params,
                 alg_params,
                 log_level='info', log_fname='log.txt', seed=None, is_continue=False, params_txt=None):
        policy_params['kwargs']['tfrecord_folders'] = alg_params['offpolicy']
        policy_params['kwargs']['batch_size'] = alg_params['batch_size']

        # labelling was done to make the tfrecords
        labeller_params = {
            'class': None,
            'kwargs': { }
        }

        # since evaluation is in tfrecords
        env_eval_params = None

        super(GCGtrainTfrecord, self).__init__(
            exp_name=exp_name,
            env_params=env_params, env_eval_params=env_eval_params,
            rp_params=rp_params, rp_eval_params=rp_eval_params,
            labeller_params=labeller_params,
            policy_params=policy_params,
            alg_params=alg_params,
            log_level=log_level, log_fname=log_fname, seed=seed, is_continue=is_continue, params_txt=params_txt
        )

        assert isinstance(self._policy, GCGPolicyTfrecord)

    ###############
    ### Restore ###
    ###############

    def _add_offpolicy(self, folders, max_to_add):
        pass # don't add to replay pools since these are tfrecords

    ########################
    ### Training methods ###
    ########################

    def _run_env_eval(self, step, do_sampler_step=True, calculate_holdout=True):
        timeit.start('eval')

        ### calculate holdout costs
        self._policy.eval_holdout()

        timeit.stop('eval')

    def _run_train_step(self, step):
        def do_train_step():
            timeit.start('train')
            self._policy.train_step(step)
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