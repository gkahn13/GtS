import os, glob, sys

from gcg.data.logger import logger

FM_DIR = os.path.realpath(os.path.dirname(__file__))
GCG_DIR = os.path.join(FM_DIR[:FM_DIR.find('gcg/src')], 'gcg')
DATA_DIR = os.path.join(GCG_DIR, 'data')
CONFIGS_DIR = os.path.join(GCG_DIR, 'configs')

class FileManager(object):

    def __init__(self, exp_name, is_continue=False, log_level='info', log_fname='log.txt', log_folder=''):
        self._exp_name = exp_name

        ### create logger
        log_path = os.path.join(self.exp_dir, log_fname)
        if not is_continue and os.path.exists(log_path):
            print('Save directory {0} exists. You need to explicitly say to continue if you want to start training '
                  'from where you left off'.format(self.exp_dir))
            sys.exit(0)

        log_folder_full = os.path.join(self.exp_dir, log_folder)
        os.makedirs(log_folder_full, exist_ok=True)
        logger.setup(display_name=self._exp_name, log_path=os.path.join(log_folder_full, log_fname), lvl=log_level)

    ####################
    ### Folder names ###
    ####################

    @property
    def exp_dir(self):
        curr_dir = os.path.dirname(__file__)
        data_dir = os.path.join(curr_dir[:curr_dir.find('src/gcg')], 'data')
        assert (os.path.exists(data_dir))
        exp_dir = os.path.join(data_dir, self._exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    ##################
    ### File names ###
    ##################

    @property
    def params_fname(self):
        return os.path.join(self.exp_dir, 'params.py')

    train_rollouts_fname_suffix = 'train_rollouts.pkl'

    def train_rollouts_fname(self, itr):
        return os.path.join(self.exp_dir, 'itr_{0:04d}_{1}'.format(itr, FileManager.train_rollouts_fname_suffix))

    eval_rollouts_fname_suffix = 'eval_rollouts.pkl'

    def eval_rollouts_fname(self, itr):
        return os.path.join(self.exp_dir, 'itr_{0:04d}_{1}'.format(itr, FileManager.eval_rollouts_fname_suffix))

    def train_policy_fname(self, itr):
        return os.path.join(self.exp_dir, 'itr_{0:04d}_train_policy.ckpt'.format(itr))

    def inference_policy_fname(self, itr):
        return os.path.join(self.exp_dir, 'itr_{0:04d}_inference_policy.ckpt'.format(itr))

    ###############
    ### Queries ###
    ###############

    def get_train_itr(self):
        train_itr = 0
        while len(glob.glob(os.path.splitext(self.inference_policy_fname(train_itr))[0] + '*')) > 0:
            train_itr += 1

        return train_itr

    def get_inference_itr(self):
        itrs = [int(f.split('_')[1]) for f in os.listdir(self.exp_dir) if self.train_rollouts_fname_suffix in f]
        inference_itr = max([-1] + itrs) + 1

        return inference_itr