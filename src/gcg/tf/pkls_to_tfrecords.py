import os
import multiprocessing

import tensorflow as tf

from gcg.data.file_manager import FileManager, DATA_DIR
from gcg.data import mypickle

from gcg.policies.gcg_policy_tfrecord import GCGPolicyTfrecord


class PklsToTfrecords(object):

    def __init__(self,
                 pkl_folders,
                 output_folder,
                 env_params,
                 labeller_params,
                 num_processes=1):
        self._pkl_folders = pkl_folders
        self._output_folder = os.path.join(DATA_DIR, output_folder)
        self._num_processes = num_processes

        ### create env
        self._env = env_params['class'](params=env_params['kwargs'])

        ### create labeller
        self._labeller = labeller_params['class'](env_spec=self._env.spec,
                                                  policy=None, # TODO
                                                  **labeller_params['kwargs']) if labeller_params['class'] else None
        if self._labeller:
            assert self._num_processes == 1, 'can only have one process when labelling'

        ### modify rollouts?
        if hasattr(self._env, 'create_rollout'):
            print('\nenv has create_rollout method')
            self._create_rollout = self._env.create_rollout
        else:
            print('\nenv does not have create_rollout method, defaulting to identity')

            def create_rollout(rollout, labeller):
                return rollout
            self._create_rollout = create_rollout

    def _convert_rollout(self, i, pkl_fname, pkl_suffix):
        rollouts = mypickle.load(pkl_fname)['rollouts']

        tfrecord_fname = os.path.join(self._output_folder, '{0:05d}_{1}.tfrecord'.format(i, pkl_suffix))

        writer = tf.python_io.TFRecordWriter(tfrecord_fname)

        for r in rollouts:

            ### modify the rollout (in case the env is different)!
            r = self._create_rollout(r, self._labeller)

            for k in ('observations_im', 'observations_vec', 'actions', 'dones', 'steps', 'goals'):
                assert k in r.keys(), '{0} not in rollout!'.format(k)

            ex = tf.train.SequenceExample()
            for k, np_dtype in zip(GCGPolicyTfrecord.tfrecord_feature_names,
                                   GCGPolicyTfrecord.tfrecord_feature_np_types):
                fl = ex.feature_lists.feature_list[k]
                for feature in r[k]:
                    fl.feature.add().bytes_list.value.append(feature.astype(np_dtype).tostring())

            writer.write(ex.SerializeToString())

        writer.close()

    def run(self):
        ### get pkl fnames
        train_pkl_fnames = []
        eval_pkl_fnames = []
        for folder in self._pkl_folders:
            folder = os.path.join(DATA_DIR, folder)
            train_pkl_fnames += [os.path.join(folder, f) for f in os.listdir(folder)
                                 if FileManager.train_rollouts_fname_suffix in f]
            eval_pkl_fnames += [os.path.join(folder, f) for f in os.listdir(folder)
                                if FileManager.eval_rollouts_fname_suffix in f]

        train_pkl_fnames = sorted(train_pkl_fnames)
        eval_pkl_fnames = sorted(eval_pkl_fnames)

        ### create output folder
        output_folder = os.path.join(DATA_DIR, self._output_folder)
        if os.path.exists(output_folder):
            assert len(os.listdir(output_folder)) == 0, '{0} output_folder is not empty'.format(output_folder)
        else:
            os.makedirs(output_folder, exist_ok=True)

        ### convert pkls to tfrecords
        for pkl_fnames, pkl_suffix in [(train_pkl_fnames, FileManager.train_rollouts_fname_suffix),
                                       (eval_pkl_fnames, FileManager.eval_rollouts_fname_suffix)]:
            pkl_suffix = os.path.splitext(pkl_suffix)[0]

            if self._num_processes > 1:
                p = multiprocessing.Pool(self._num_processes)
                p.starmap(self._convert_rollout,
                          [(i, pkl_fname, pkl_suffix) for i, pkl_fname in enumerate(pkl_fnames)])
            else:
                for i, pkl_fname in enumerate(pkl_fnames):
                    # print(pkl_fnames)
                    self._convert_rollout(i, pkl_fname, pkl_suffix)

