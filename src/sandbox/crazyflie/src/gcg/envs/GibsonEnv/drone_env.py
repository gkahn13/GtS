from sandbox.crazyflie.src.gcg.envs.GibsonEnv.env_modalities import CameraRobotEnv, BaseRobotEnv
from sandbox.crazyflie.src.gcg.envs.GibsonEnv.env_bases import *
from sandbox.crazyflie.src.gcg.envs.GibsonEnv.robot_locomotors import Quadrotor3
from transforms3d import quaternions
import os
import numpy as np
import sys
import pybullet as p
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data
import cv2
import random
from gcg.envs.env_spec import EnvSpec
from collections import OrderedDict
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from termcolor import colored

CALC_OBSTACLE_PENALTY = 1

tracking_camera = {
    'yaw': 20,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

tracking_camera_top = {
    'yaw': 20,  # demo: living room, stairs
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}
 
class DroneNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_count=0):
        #self.config = self.parse_config(config)
        self.config = config

        
        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Quadrotor3(self.config, env=self))
        self.scene_introduce()
        self.gui = self.config["mode"] == "gui"
        self.total_reward = 0
        self.total_frame = 0
        


    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x,y,z = self.robot.body_xyz
        r,p,ya = self.robot.body_rpy
        cv2.putText(img, 'x:{0:.4f} y:{1:.4f} z:{2:.4f}'.format(x,y,z), (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r,p,ya), (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'fps:{0:.4f}'.format(self.fps), (10, 80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            # print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        # print(self.robot.feet_contact)

        #electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we 
        electricity_cost  = self.stall_torque_cost * float(np.square(a).mean())


        debugmode = 0
        wall_contact = [pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_goal = 0
        if self.robot.dist_to_target() < 2:
            close_to_goal = 0.5

        obstacle_penalty = 0

        debugmode = 0

        debugmode = 0
        if (debugmode):
            print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            print("electricity_cost", electricity_cost)
            print("close to goal", close_to_goal)
            #print("progress")
            #print(progress)
            #print("electricity_cost")
            #print(electricity_cost)
            #print("joints_at_limit_cost")
            #print(joints_at_limit_cost)
            #print("feet_collision_cost")
            #print(feet_collision_cost)

        rewards = [
            #alive,
            progress,
            #wall_collision_cost,
            close_to_goal,
            obstacle_penalty
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):

        done = self.nframe > 250 or self.robot.get_position()[2] < 0
        return done

    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0

        obs = CameraRobotEnv._reset(self)
        return obs


class GcgDroneNavigateEnv(DroneNavigateEnv):
    def __init__(self, params={}, gpu_count=0):
        DroneNavigateEnv.__init__(self, params, gpu_count)

        self._obs_shape = params['obs_shape']
        self._yaw_limits = params['yaw_limits']

        self._horizon = params['horizon']
        self._model_id = params['model_id']
        self._setup_spec()
        assert (self.observation_im_space.shape[-1] == 1 or self.observation_im_space.shape[-1] == 3)
        self.spec = EnvSpec(self.observation_im_space,self.action_space,self.action_selection_space,self.observation_vec_spec,self.action_spec,self.action_selection_spec,self.goal_spec)

    @property
    def horizon(self):
        return self._horizon
    

    def _setup_spec(self):
        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()

        self.action_spec['yaw'] = Box(low=-180, high=180)

        self.action_space = Box(low=np.array([self.action_spec['yaw'].low[0]]),
                                high=np.array([self.action_spec['yaw'].high[0]]))

        self.action_selection_spec['yaw'] = Box(low=self._yaw_limits[0], high=self._yaw_limits[1])

        self.action_selection_space = Box(low = np.array([self.action_selection_spec['yaw'].low[0]]), high = np.array([self.action_selection_spec['yaw'].high[0]]))
        assert (np.logical_and(self.action_selection_space.low >= self.action_space.low,
                               self.action_selection_space.high <= self.action_space.high).all())

        self.observation_im_space = Box(low=0, high=255, shape=self._obs_shape)
        self.observation_vec_spec['coll'] = Discrete(1)

    def step(self, a):
        observations, reward, _, env_info_internal = DroneNavigateEnv._step(self, a)
        done = self.get_collision()
        filtered_obs = self.get_filtered_observation(observations)
        env_info = dict(x=env_info_internal["x"], y=env_info_internal["y"], yaw=env_info_internal["yaw"], height=env_info_internal["height"], speed=env_info_internal["speed"], model_id=self._model_id)
        return filtered_obs, np.array([]), reward, done, env_info

    def reset(self, offline=False, keep_rosbag=True):
        observations =  DroneNavigateEnv._reset(self)
        filtered_obs = self.get_filtered_observation_reset(observations)

        return filtered_obs, np.array([])

    def ros_is_good(self, print=False):
        return True

    def get_collision(self):
        collision = (len(self.robot.parts['base_link'].contact_list())  > 0) or (abs(self.robot.get_orientation_eulerian()[0]) > 0.5) or (abs(self.robot.get_orientation_eulerian()[1]) > 0.5)
        if collision:
            print("\n")
            print(colored("COLLISION!!!!!", "green"))
            print("\n")
            print(colored("COLLISION!!!!!", "red"))
            print("\n")
            print(colored("COLLISION!!!!!", "yellow"))
            print("\n")
        return collision

    def get_filtered_observation(self, observations):
        image = observations['rgb_filled']
        image_resized = cv2.cvtColor(cv2.resize(image, (96, 96))[12:84], cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', image_resized)
        cv2.waitKey(5)
        return (image_resized, np.array([int(self.get_collision())]))

    def get_filtered_observation_reset(self, observations):
        image = observations['rgb_filled']
        image_resized = cv2.cvtColor(cv2.resize(image, (96, 96))[12:84], cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', image_resized)
        cv2.waitKey(5)
        return (image_resized, np.array([0]))
