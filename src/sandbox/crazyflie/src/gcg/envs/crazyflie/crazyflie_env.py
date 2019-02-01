import os
from collections import OrderedDict
import numpy as np
from PIL import Image
from io import BytesIO
from gcg.envs.env import Env
from gcg.envs.env_spec import EnvSpec
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.data.logger import logger
import matplotlib.pyplot as plt
import threading
import time
import rospy
import rosbag
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import crazyflie.msg


class RolloutRosbag:
    def __init__(self):
        self._rosbag = None
        self._last_write = None

    @property
    def _rosbag_dir(self):
        dir = os.path.join(logger.dir, 'rosbags')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    def _rosbag_name(self, num):
        return os.path.join(self._rosbag_dir, 'rosbag{0:04d}.bag'.format(num))

    @property
    def is_open(self):
        return (self._rosbag is not None)

    def open(self):
        assert (not self.is_open)

        bag_num = 0
        while os.path.exists(self._rosbag_name(bag_num)):
            bag_num += 1

        self._rosbag = rosbag.Bag(self._rosbag_name(bag_num), 'w', compression='bz2')
        self._last_write = rospy.Time.now()

    def write(self, topic, msg, stamp):
        assert (self._rosbag is not None)

        if msg is not None and stamp is not None:
            if stamp > self._last_write:
                self._rosbag.write(topic, msg)
        else:
            logger.warn('Topic {0} not received'.format(topic))

    def write_all(self, topics, msg_dict, stamp_dict):
        for topic in topics:
            self.write(topic, msg_dict.get(topic), stamp_dict.get(topic))

    def close(self):
        assert (self.is_open)

        self._rosbag.close()
        self._rosbag = None
        self._last_write = None

    def trash(self):
        assert (self.is_open)
        bag_fname = self._rosbag.filename
        try:
            self.close()
        except:
            pass

        os.remove(os.path.join(self._rosbag_dir, bag_fname))


class CrazyflieEnv(Env):
    def __init__(self, params={}):
        params.setdefault('dt', 0.25)
        params.setdefault('horizon', int(5. * 60. / params['dt']))  # 5 minutes worth
        params.setdefault('ros_namespace', '/crazyflie/')
        params.setdefault('obs_shape', (72, 96, 1))
        params.setdefault('yaw_limits', [-120, 120]) #default yaw rate range
        params.setdefault('fixed_alt', 0.4)
        params.setdefault('fixed_velocity_range', [0.4, 0.4])
        params.setdefault('press_enter_on_reset', False)
        params.setdefault('prompt_save_rollout_on_coll', False)
        params.setdefault('enable_adjustment_on_start', True)
        params.setdefault('use_joy_commands', True)
        params.setdefault('joy_start_btn', 1) #A
        params.setdefault('joy_stop_btn', 2) #B
        params.setdefault('joy_coll_stop_btn', 0) #X
        params.setdefault('joy_trash_rollout_btn', 3) # Y
        params.setdefault('joy_topic', '/joy')
        params.setdefault('collision_reward', 1)
        params.setdefault('collision_reward_only', True)

        self._obs_shape = params['obs_shape']
        self._yaw_limits = params['yaw_limits']
        self._fixed_alt = params['fixed_alt']
        self._collision_reward = params['collision_reward']
        self._collision_reward_only = params['collision_reward_only']
        self._fixed_velocity_range = params['fixed_velocity_range']
        self._fixed_velocity = np.random.uniform(self._fixed_velocity_range[0], self._fixed_velocity_range[1])
        self._dt = params['dt']
        self.horizon = params['horizon']

        # start stop and pause
        self._enable_adjustment_on_start = params['enable_adjustment_on_start']
        self._use_joy_commands = params['use_joy_commands']
        self._joy_topic = params['joy_topic']
        self._joy_stop_btn = params['joy_stop_btn']
        self._joy_coll_stop_btn = params['joy_coll_stop_btn']
        self._joy_start_btn = params['joy_start_btn']
        self._joy_trash_rollout_btn = params['joy_trash_rollout_btn']
        self._press_enter_on_reset = params['press_enter_on_reset']
        self._prompt_save_rollout_on_coll = params['prompt_save_rollout_on_coll']
        self._start_pressed = False
        self._stop_pressed = False
        self._trash_rollout = False
        self._coll_stop_pressed = False
        self._curr_joy = None
        self._curr_motion = crazyflie.msg.CFMotion()

        self._setup_spec()
        assert (self.observation_im_space.shape[-1] == 1 or self.observation_im_space.shape[-1] == 3)
        self.spec = EnvSpec(
            observation_im_space=self.observation_im_space,
            action_space=self.action_space,
            action_selection_space=self.action_selection_space,
            observation_vec_spec=self.observation_vec_spec,
            action_spec=self.action_spec,
            action_selection_spec=self.action_selection_spec,
            goal_spec=self.goal_spec)

        self._last_step_time = None
        self._is_collision = False

        rospy.init_node('CrazyflieEnv', anonymous=True)
        time.sleep(0.5)

        self._ros_namespace = params['ros_namespace']
        self._ros_topics_and_types = dict([
            ('cf/0/image', sensor_msgs.msg.CompressedImage),
            ('cf/0/data', crazyflie.msg.CFData),
            ('cf/0/coll', std_msgs.msg.Bool),
            ('cf/0/motion', crazyflie.msg.CFMotion)

        ])
        self._ros_msgs = dict()
        self._ros_msg_times = dict()
        for topic, type in self._ros_topics_and_types.items():
            rospy.Subscriber(topic, type, self.ros_msg_update, (topic,))
        
        self._ros_motion_pub = rospy.Publisher("/cf/0/motion", crazyflie.msg.CFMotion, queue_size=10)
        self._ros_command_pub = rospy.Publisher("/cf/0/command", crazyflie.msg.CFCommand, queue_size=10)
        self._ros_stop_pub = rospy.Publisher('/joystop', crazyflie.msg.JoyStop, queue_size=10)
        if self._use_joy_commands:
            logger.debug("Environment using joystick commands")
            self._ros_joy_sub = rospy.Subscriber(self._joy_topic, sensor_msgs.msg.Joy, self._joy_cb)



        # I don't think this is needed
        # self._ros_pid_enable_pub = rospy.Publisher(self._ros_namespace + 'pid/enable', std_msgs.msg.Empty,
        #                                            queue_size=10)
        # self._ros_pid_disable_pub = rospy.Publisher(self._ros_namespace + 'pid/disable', std_msgs.msg.Empty,
        #                                             queue_size=10)

        self._ros_rolloutbag = RolloutRosbag()
        self._t = 0


        self.suppress_output = False
        self.resetting = False
        self._send_override = False # set true only when resetting but still wanting to send background thread motion commands
        threading.Thread(target = self._background_thread).start()

        time.sleep(1.0) #waiting for some messages before resetting

        self.delete_this_variable = 0
        
    def _setup_spec(self):
        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()

        self.action_spec['yaw'] = Box(low=-180, high=180)
        self.action_space = Box(low=np.array([self.action_spec['yaw'].low[0]]), high = np.array([self.action_spec['yaw'].high[0]]))
        self.action_selection_spec['yaw'] = Box(low=self._yaw_limits[0], high=self._yaw_limits[1])
        self.action_selection_space = Box(low=np.array([self.action_selection_spec['yaw'].low[0]]), high = np.array([self.action_selection_spec['yaw'].high[0]]))

        assert (np.logical_and(self.action_selection_space.low >= self.action_space.low,
                               self.action_selection_space.high <= self.action_space.high).all())

        self.observation_im_space = Box(low=0, high=255, shape=self._obs_shape)
        self.observation_vec_spec['coll'] = Discrete(1)


    def _log(self, msg, lvl):
        if not self.suppress_output:
            if lvl == "info":
                logger.info(msg)
            elif lvl == "debug":
                logger.debug(msg)
            elif lvl == "warn":
                logger.warn(msg)
            elif lvl == "error":
                logger.error(msg)
            else:
                print("NOT VALID LOG LEVEL")


    def _get_observation(self):
        msg = self._ros_msgs['cf/0/image']

        recon_pil_jpg = BytesIO(msg.data)
        recon_pil_arr = Image.open(recon_pil_jpg)

        is_grayscale = (self.observation_im_space.shape[-1] == 1)
        if is_grayscale:
            grayscale = recon_pil_arr.convert('L')
            grayscale_resized = grayscale.resize(self.observation_im_space.shape[:-1][::-1],
                                                 Image.ANTIALIAS)  # b/c (width, height)
            im = np.expand_dims(np.array(grayscale_resized), 2)
        else:
            rgb_resized = recon_pil_arr.resize(self.observation_im_space.shape[:-1][::-1],
                                               Image.ANTIALIAS)  # b/c (width, height)
            im = np.array(rgb_resized)

        coll = self._get_collision() or self._get_joy_coll_stop()
        vec = np.array([coll])

        if(vec[0]==1):
            self._log("COLLISION!!", "info")
        return im, vec

    def _get_goal(self):
        return np.array([])

    def _get_reward(self):
        if self._is_collision:
            reward = self._collision_reward
        else:
            reward = 0
        return reward

    ### JOYSTICK ###

    def _get_joy_stop(self):
        #in this order
        return self._use_joy_commands and self._stop_pressed

    def _get_joy_coll_stop(self):
        return self._use_joy_commands and self._coll_stop_pressed

    def _get_trash_rollout(self):
        return self._use_joy_commands and self._trash_rollout

    def _get_joy_start(self):
        #in this order
        return self._use_joy_commands and self._start_pressed
        self._curr_joy and self._curr_joy.buttons[self._joy_start_btn]

    ### OTHER ###
    
    def _get_collision(self):
        return self._is_collision 

    def _get_done(self):
        return self._get_joy_stop() or self._get_collision() or self._get_trash_rollout() or self._get_joy_coll_stop()

    def _background_thread(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if (self._use_joy_commands and not self.resetting) or self._send_override:
                self._ros_motion_pub.publish(self._curr_motion)
            rate.sleep()

    def _set_motion(self, x, y, yaw, dz):
        motion = crazyflie.msg.CFMotion()
        motion.x = x
        motion.y = y
        motion.yaw = yaw
        motion.dz = dz
        motion.is_flow_motion = True
        self._curr_motion = motion

    def _set_command(self, cmd):
        #0 is ESTOP, 1 IS LAND, 2 IS TAKEOFF
        command = crazyflie.msg.CFCommand()
        command.cmd = cmd
        self._ros_command_pub.publish(command)

    def _set_stop(self, msg):
        #0 is no coll stop, 1 is coll stop
        stop = crazyflie.msg.JoyStop() 
        stop.stop = msg
        self._ros_stop_pub.publish(stop)

    def _joy_cb(self, msg):
        self._curr_joy = msg
        #permanent state
        if self._curr_joy.buttons[self._joy_start_btn]:
            self._start_pressed = True

        if self._curr_joy.buttons[self._joy_stop_btn]:
            self._stop_pressed = True
            self._set_stop(0)

        if self._curr_joy.buttons[self._joy_coll_stop_btn]:
            self._coll_stop_pressed = True
            self._set_stop(1)

        if self._curr_joy.buttons[self._joy_trash_rollout_btn]:
            self._trash_rollout = True

    def step(self, action, offline=False):
        if not offline:
            assert (self.ros_is_good())

        if not self._ros_rolloutbag.is_open and self._get_joy_start():
            self._log("Beginning Episode...", "info")
            self._ros_rolloutbag.open()
        elif not self._ros_rolloutbag.is_open:
            self._log("Waiting for Joystick input ...", "debug")

        action = np.asarray(action)
        if not (np.logical_and(action >= self.action_space.low, action <= self.action_space.high).all()):
            self._log('Action {0} will be clipped to beget_obser within bounds: {1}, {2}'.format(action,
                                                                                          self.action_space.low,
                                                                                          self.action_space.high), "warn")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        if self._get_joy_coll_stop() or self._get_trash_rollout():
            self._set_motion(0,0,0,0)
        else:
            yaw = action[0]
            self._set_motion(self._fixed_velocity, 0, yaw, 0)

        if not offline:
            rospy.sleep(max(0., self._dt - (rospy.Time.now() - self._last_step_time).to_sec()))
            self._last_step_time = rospy.Time.now()

        done = self._get_done()
        if done:
            self._log('Done after {0} steps'.format(self._t), "warn")
            if self._get_collision():
                self._log('-- COLLISION --', "warn")
            elif self._get_joy_stop() or self._get_joy_coll_stop():
                self._log('-- MANUALLY STOPPED --', "warn")
            elif self._get_trash_rollout():
                self._log('-- TRASHING ROLLOUT --', "warn")
            else:
                self._log('-- DONE for unknown reason --', "warn")

        if not offline and self._ros_rolloutbag.is_open:
            #change last collision msg to have a 1 if manually noted a collision
            if self._get_joy_coll_stop():
                self._ros_msgs[[topic for topic in self._ros_msgs if 'coll' in topic][0]].data = 1
            self._ros_rolloutbag.write_all(self._ros_topics_and_types.keys(), self._ros_msgs, self._ros_msg_times)
            if done:
                self._log('Done after {0} steps'.format(self._t), "debug")
                self._t = 0

        next_observation = self._get_observation()
        goal = self._get_goal()
        reward = self._get_reward()
        env_info = dict()
        self._t += 1

        return next_observation, goal, reward, done, env_info


    def reset_state(self):
        self._last_step_time = rospy.Time.now()
        self._is_collision = False
        self._t = 0
        self._done = False
        self._curr_joy = None
        self._start_pressed = False
        self._stop_pressed = False
        self._trash_rollout = False
        self._coll_stop_pressed = False
        self._send_override = False
        self._fixed_velocity = np.random.uniform(self._fixed_velocity_range[0], self._fixed_velocity_range[1])
        self._log('Velocity: {0}'.format(self._fixed_velocity), "debug")


    def reset(self, offline=False, keep_rosbag=True):
        self.resetting = True
        if offline:
            self._is_collision = False
            return self._get_observation(), self._get_goal()
        assert self.ros_is_good(), "On End: ROS IS NOT GOOD"

        manual_collision_label= self._get_joy_coll_stop()
        trash_rollout = self._get_trash_rollout()
        
        
        if self._ros_rolloutbag.is_open and (trash_rollout or not keep_rosbag or self._get_joy_stop()):
            # should've been closed in step when done
            self._log('Trashing bag', "debug")
            self._ros_rolloutbag.trash()

        if self._is_collision:
            self._log('Resetting (collision)', "debug")
        elif self._get_joy_stop():
            self._log('Resetting (joy stop)', "debug")
        elif self._get_joy_coll_stop():
            self._log('Resetting (joy collision stop)', "debug")
            # if manual collision label pressed, start new episode, but skip landing and takeoff
        elif self._get_trash_rollout():
            self._log('Resetting (trash rollout)', "debug")
        else:
            self._log('Resetting (other)', "debug")

        # end behavior
        if not manual_collision_label and not trash_rollout:
            self._log("Landing.", 'debug')
            self._set_command(0) # land
            rospy.sleep(2.0)
        else:
            self._send_override = True
            self._set_motion(0,0,0,0) # stay hovering
            time.sleep(1.0) # stopping motion

        # if its still open, that means it wasn't flagged to be trashed
        # this happens after landing to save battery
        if self._ros_rolloutbag.is_open:
            #if is collision and config says we need to save, then save on user input, else just save always
            if self._is_collision and self._prompt_save_rollout_on_coll:
                self._log("Real Collision Happened. Save Rollout? (y/n) default [y]: ", "warn")
                res = input()
                if res is '' or res is 'y':
                    self._log('Saving bag', "debug")
                    self._ros_rolloutbag.close()
                else:            
                    self._log('Trashing bag', "debug")
                    self._ros_rolloutbag.trash()
            else:
                self._ros_rolloutbag.close()

        # other cases
        if self._press_enter_on_reset:
            self._log('Resetting, press enter to continue', "info")
            input()
        elif not manual_collision_label and not trash_rollout and self.is_upside_down():
            self._log('Crazyflie is upside down', "info")
            self._log('Resetting, press enter to continue', "info")
            input()

        self.reset_state()

        assert self.ros_is_good() # "On Start: ROS IS NOT GOOD"

        if not manual_collision_label and not trash_rollout:
            self._log("Waiting to takeoff ...", "debug")
            rospy.sleep(1.0)

            self._log("Taking off", "debug")
            self._set_command(2)
            rospy.sleep(2.0)


        #readjustment means full joystick control until you press A (this replaces Wait For Start)
        if self._use_joy_commands and self._enable_adjustment_on_start:
            #orientation adjustment mode
            self._send_override = True
            self._log("Starting Position Adjustment Mode. Press Start Button to proceed.", 'info')
            self._set_motion(0,0,0,0) # full stop
            while not self._get_joy_start():
                if self._use_joy_commands and self._curr_joy:
                    # use yaw axis, and vx axis
                    self._set_motion(self._curr_joy.axes[3] * 0.5, 0, self._curr_joy.axes[0] * -120, 0)
            
            self._set_motion(0,0,0,0) # full stop

            #once more to reset any other variables that may have changed
            self.reset_state();

            self._log("Exiting Position Mode.", 'info')
            self._log("Beginning Episode...", "info")
            self._ros_rolloutbag.open()

        self.resetting = False

        if not self._use_joy_commands:
            self._log("Beginning Episode...", "info")
            self._ros_rolloutbag.open()

        return self._get_observation(), self._get_goal()

    ###########
    ### ROS ###
    ###########

    def ros_msg_update(self, msg, args):
        topic = args[0]


        if 'coll' in topic:
            if msg.data == 1:
                self._is_collision = False

            if self._is_collision:
                if msg.data == 1:
                    # if is_collision and current is collision, update
                    self._ros_msgs[topic] = msg
                    self._ros_msg_times[topic] = rospy.Time.now()
                else:
                    if self._ros_msgs[topic].data != 1:
                        # if is collision, but previous message is not collision, then this topic didn't cause a colision
                        self._ros_msgs[topic] = msg
                        self._ros_msg_times[topic] = rospy.Time.now()
            else:
                # always update if not in collision
                self._ros_msgs[topic] = msg
                self._ros_msg_times[topic] = rospy.Time.now()

        elif 'stop' in topic:
            if msg.stop == 1:
                self._is_collision = True

            if self._is_collision:
                if msg.stop == 1:
                    # if is_collision and current is collision, update
                    self._ros_msgs[topic] = msg
                    self._ros_msg_times[topic] = rospy.Time.now()
                else:
                    if self._ros_msgs[topic].data != 1:
                        # if is collision, but previous message is not collision, then this topic didn't cause a colision
                        self._ros_msgs[topic] = msg
                        self._ros_msg_times[topic] = rospy.Time.now()
            else:
                # always update if not in collision
                self._ros_msgs[topic] = msg
                self._ros_msg_times[topic] = rospy.Time.now()
        else:
            #CF data unpacking
            # if 'data' in topic:
            #     self._ros_msgs['accel_x'] = msg.accel_x
            #     self._ros_msg_times['accel_x'] = rospy.Time.now()
            #     self._ros_msgs['accel_y'] = msg.accel_y
            #     self._ros_msg_times['accel_y'] = rospy.Time.now()
            #     self._ros_msgs['accel_z'] = msg.accel_z
            #     self._ros_msg_times['accel_z'] = rospy.Time.now()
            #     self._ros_msgs['alt'] = msg.alt
            #     self._ros_msg_times['alt'] = rospy.Time.now()
            #     self._ros_msgs['v_batt'] = msg.v_batt
            #     self._ros_msg_times['v_batt'] = rospy.Time.now()
            # else:
            self._ros_msgs[topic] = msg
            self._ros_msg_times[topic] = rospy.Time.now()

    def ros_is_good(self, print=True):
        for topic in self._ros_topics_and_types.keys():
            if 'cmd' not in topic and 'coll' not in topic and 'motion' not in topic and 'stop' not in topic:
                if topic not in self._ros_msg_times:
                    if print:
                        self._log('Topic {0} has never been received'.format(topic), "debug")
                    return False
                elapsed = (rospy.Time.now() - self._ros_msg_times[topic]).to_sec()
                if elapsed > self._dt * 50:
                    if print:
                        self._log('Topic {0} was received {1} seconds ago (dt is {2})'.format(topic, elapsed, self._dt), "debug")
                    return False
        return True

    def is_upside_down(self):
        alt = self._ros_msgs['cf/0/data'].alt
        if (alt > 0.05):
            return True
        return False

