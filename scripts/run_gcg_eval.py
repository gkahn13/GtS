import os
import argparse

from gcg.misc.utils import import_params
from gcg.algos.gcg_eval import GCGeval

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('-itr', type=int)
args = parser.parse_args()

# load config
py_config_path = os.path.abspath('../configs/{0}.py'.format(args.exp))
assert(os.path.exists(py_config_path))
params = import_params(py_config_path)
with open(py_config_path, 'r') as f:
    params_txt = ''.join(f.readlines())

# create algorithm
AlgoClass = params['eval']['class']
assert(issubclass(AlgoClass, GCGeval))
algo = AlgoClass(eval_itr=args.itr,
                 eval_params=params['eval']['kwargs'],
                 exp_name=params['exp_name'],
                 env_eval_params=params['env_eval'],
                 policy_params=params['policy'],
                 rp_eval_params=params['replay_pool_eval'],
                 seed=params['seed'],
                 log_level=params['log_level'])

# run algorithm
algo.run()
