import os, sys, glob

import numpy as np

from gcg.algos.gcg import GCG
from gcg.policies.gcg_policy import GCGPolicy
from gcg.samplers.sampler import Sampler
from gcg.data.logger import logger
from gcg.data import mypickle

from gcg.algos.gcg_eval import GCGeval, run_eval_gcg

class EvalCrazyflieGCG(GCGeval):

    def __init__(self,
                 eval_itr,
                 eval_params,
                 exp_name,
                 env_eval_params,
                 policy_params,
                 rp_eval_params,
                 seed=None, log_level='info',  log_fname='log_eval.txt'):
        GCGeval.__init__(self,
                 eval_itr,
                 eval_params,
                 exp_name,
                 env_eval_params,
                 policy_params,
                 rp_eval_params,
                 seed=None, log_level='info',  log_fname='log_eval.txt')

    ############
    ### Eval ###
    ############

    def _eval_reset(self, **kwargs):
        while True:
            try:
                self._sampler.reset(**kwargs)
                break
            except Exception as e:
                logger.warn('Reset exception reset{0}'.format(str(e)))
                logger.info('Press enter to continue')
                input()
                logger.info('')
        
    def _eval_step(self):
        try:
            self._sampler.step(step=0,
                               take_random_actions=False,
                               explore=False)
        except Exception as e:
            logger.warn('Sampler exception step{0}'.format(str(e)))
            self._sampler.trash_current_rollouts()

            logger.info('Press enter to continue')
            input()
            self._eval_reset(keep_rosbag=False)

    def _eval_save(self, rollouts, new_rollouts):
        logger.info('')
        logger.info('Keep rollout?')
        response = input()
        if response != 'y':
            logger.info('NOT saving rollouts')
        else:
            logger.info('Saving rollouts')
            rollouts += new_rollouts
            self._save_eval_rollouts(rollouts)

        return rollouts
    
def eval_crazyflie_gcg(params, itr):
    run_eval_gcg(params, itr, EvalClass=EvalCrazyflieGCG)
