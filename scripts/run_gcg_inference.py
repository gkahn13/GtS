import os
import argparse

from gcg.misc.utils import import_params
from gcg.algos.gcg_inference import GCGinference

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('--continue', action='store_true')
args = parser.parse_args()

# load config
py_config_path = os.path.abspath('../configs/{0}.py'.format(args.exp))
assert(os.path.exists(py_config_path))
params = import_params(py_config_path)
with open(py_config_path, 'r') as f:
    params_txt = ''.join(f.readlines())

# create algorithm
AlgoClass = params['alg']['class']
assert(issubclass(AlgoClass, GCGinference))
algo = AlgoClass(exp_name=params['exp_name'],
                 env_params=params['env'],
                 env_eval_params=params['env_eval'],
                 rp_params=params['replay_pool'],
                 rp_eval_params=params['replay_pool_eval'],
                 labeller_params=params['labeller'],
                 policy_params=params['policy'],
                 alg_params=params['alg']['kwargs'],
                 log_level=params['log_level'],
                 log_fname='log_inference.txt',
                 is_continue=getattr(args, 'continue'),
                 params_txt=params_txt,
                 seed=params['seed'])

# run algorithm
algo.run()
