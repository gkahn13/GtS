import os
import argparse

from gcg.misc.utils import import_params
#from gcg.data.file_manager import CONFIGS_DIR

from gcg.tf.pkls_to_tfrecords import PklsToTfrecords

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str, help='which experiment are you creating the tfrecords for?')
parser.add_argument('-pkl_folders', nargs='+', help='list of folders containing pkls (w.r.t. data folder path)')
parser.add_argument('-output_folder', type=str, help='where to save the files')
parser.add_argument('--num_processes', type=int, default=1, help='number of processes to use. can only be > 1 if no labeller')
args = parser.parse_args()

# load config
py_config_path = os.path.abspath('../configs/{0}.py'.format(args.exp))
assert(os.path.exists(py_config_path))
params = import_params(py_config_path)

# create class
pkl_to_tf = PklsToTfrecords(pkl_folders=args.pkl_folders,
                            output_folder=args.output_folder,
                            env_params=params['env'],
                            labeller_params=params['labeller'],
                            num_processes=args.num_processes)

# run conversion
pkl_to_tf.run()
