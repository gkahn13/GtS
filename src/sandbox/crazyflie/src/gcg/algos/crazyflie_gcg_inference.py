import os, time
from collections import defaultdict

import numpy as np

import rosbag

from gcg.algos.gcg_inference import GCGinference
from gcg.data.logger import logger
from gcg.data.timer import timeit



class CrazyflieGCGinference(GCGinference):
    def __init__(self,
                 exp_name,
                 env_params, env_eval_params,
                 rp_params, rp_eval_params,
                 labeller_params,
                 policy_params,
                 alg_params,
                 log_level='info', log_fname='log.txt', seed=None, is_continue=False, params_txt=None):

        self._train_holdout_pct = alg_params['train_holdout_pct']
        self._added_rosbag_filenames = []

        super(CrazyflieGCGinference, self).__init__(
            exp_name=exp_name,
            env_params=env_params, env_eval_params=env_eval_params,
            rp_params=rp_params, rp_eval_params=rp_eval_params,
            labeller_params=labeller_params,
            policy_params=policy_params,
            alg_params=alg_params,
            log_level=log_level, log_fname=log_fname, seed=seed, is_continue=is_continue, params_txt=params_txt
        )


    #############
    ### Files ###
    #############

    @property
    def _rosbag_dir(self):
        return os.path.join(self._fm.exp_dir, 'rosbags')

    def _rosbag_file_name(self, num):
        return os.path.join(self._rosbag_dir, 'rosbag{0:04d}.bag'.format(num))

    ###############
    ### Restore ###
    ###############

    def _split_rollouts(self, filenames):
        ### ensure that the training and holdout sets are always split the same
        np_random_state = np.random.get_state()
        np.random.seed(0)

        train_rollouts = []
        holdout_rollouts = []
        for fname in filenames:
            if np.random.random() > self._train_holdout_pct:
                train_rollouts.append(fname)
            else:
                holdout_rollouts.append(fname)

        np.random.set_state(np_random_state)

        return sorted(train_rollouts), sorted(holdout_rollouts)

    def _add_rosbags(self, sampler, rosbag_filenames):

        timesteps_kept = 0
        timesteps_total = 0
        for fname in rosbag_filenames:
            self._added_rosbag_filenames.append(fname)

            ### read bag file
            try:
                bag = rosbag.Bag(fname, 'r', compression='bz2')
            except Exception as e:
                logger.warn('{0}: could not open'.format(os.path.basename(fname)))
                print(e)
                continue
            d_bag = defaultdict(list)

            # bag.read_messages
            for topic, msg, t in bag.read_messages():
                if topic == 'joystop' and msg.stop == 1:
                    logger.warn('{0}: has incorrect collision detection. Skipping.'.format(os.path.basename(fname)))
                    continue
                elif topic == 'joystop' and msg.stop == 0:
                    break
                else:
                    d_bag[topic].append(msg)

            bag.close()

            if len(d_bag['cf/0/data']) == 0:
                logger.warn('{0}: has no entries. Skipping.'.format(os.path.basename(fname)))
                continue

            timesteps_total += len(d_bag['cf/0/data']) - 1

            d_bag_parsed = defaultdict(list)
            time_of_coll = 0

            for t in range(min([len(d_bag['cf/0/data']), len(d_bag['cf/0/motion']), len(d_bag['cf/0/coll']), len(d_bag['cf/0/image'])])):
                for topic in d_bag:
                    try:
                        d_bag_parsed[topic].append(d_bag[topic][t])
                    except:
                        import IPython; IPython.embed()
                if d_bag['cf/0/coll'][t].data == 1:
                    time_of_coll = t
                    break

            parsed_colls = np.array([msg.data for msg in d_bag_parsed['cf/0/coll']])
            colls = np.array([msg.data for msg in d_bag['cf/0/coll']])
            
            if len(parsed_colls) < 10:
                logger.warn('{0}: had a collision too early: at timestep {1}. Skipping'.format(os.path.basename(fname), len(parsed_colls)))
                continue

            logger.info('Added rosbag: {0}, with {1}/{2} timesteps'.format(os.path.basename(fname), len(parsed_colls), len(colls)))
            timesteps_kept += len(parsed_colls) - 1
            
            ### update env and step
            def update_env(t):
                for key in d_bag.keys():
                    try:
                        sampler.env.ros_msg_update(d_bag[key][t], [key])
                    except Exception as e:
                        print("Issue updating env: ", str(e))
                        import IPython; IPython.embed()

            #makes sure no statements are printed
            self._env.suppress_output = True

            update_env(0)
            if len(sampler) == 0:
                logger.warn('Resetting!')
                sampler.reset(offline=True)

            bag_length = min([len(d_bag['cf/0/data']), len(d_bag['cf/0/motion']), len(d_bag['cf/0/coll']), len(d_bag['cf/0/image'])])
            for t in range(1, bag_length):
                update_env(t)
                motion = d_bag['cf/0/motion'][t-1]  
                action = np.array([motion.x, motion.y, motion.yaw, motion.dz])
                st = time.time()
                sampler.step(len(sampler), action=action, offline=True)
                endt = time.time()

            self._env.suppress_output = False

            if not sampler.is_done_nexts:
                logger.warn('{0}: did not end in done, manually resetting'.format(os.path.basename(fname)))
                sampler.reset(offline=True)

            num_steps = len(d_bag['mode']) - 1 
            timesteps_kept += num_steps if num_steps > 0 else 0

        logger.info('Adding {0:d} timesteps ({1:.2f} kept)'.format(timesteps_kept, timesteps_kept / float(timesteps_total+1)))


    def _add_offpolicy(self, folders, max_to_add):
        for folder in folders:
            assert (os.path.exists(folder), 'offpolicy folder {0} does not exist'.format(folder))
            logger.info('Loading rosbag data from {0}'.format(folder))
            rosbag_filenames = sorted(
                [os.path.join(folder, fname) for fname in os.listdir(folder) if '.bag' in fname])
            train_rosbag_filenames, holdout_rosbag_filenames = self._split_rollouts(rosbag_filenames)
            logger.info('Adding train...')
            self._add_rosbags(self._sampler, self._replay_pool, train_rosbag_filenames)
            logger.info('Adding holdout...')
            self._add_rosbags(self._sampler_eval, self._replay_pool_eval, holdout_rosbag_filenames)
        logger.info('Added {0} train samples'.format(len(self._replay_pool)))
        logger.info('Added {0} holdout samples'.format(len(self._replay_pool_eval)))

    ########################
    ### Training methods ###
    ########################

    def _run_env_step(self, step):
        timeit.start('sample')
        self._sampler.step(step,
                           take_random_actions=(step < self._onpolicy_after_n_steps),
                           explore=True)
        timeit.stop('sample')

        return step

    def _run_reset_sampler(self, keep_rosbag=True):
        while True:
            try:
                self._sampler.reset(keep_rosbag=keep_rosbag)
                break
            except Exception as e:
                logger.warn('Reset exception {0}'.format(str(e)))
                while not self._env.ros_is_good(print=False):
                    time.sleep(0.25)
                logger.warn('Continuing...')
