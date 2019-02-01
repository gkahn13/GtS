from darkflow.net.build import TFNet
import os
import numpy as np
from pathlib import Path

from gcg.labellers.labeller import Labeller

class FoodLabeller(Labeller):

    def __init__(self, env_spec, policy, options=None, darkflow_path=None, im_is_rgb=False, **kwargs):
        super(FoodLabeller, self).__init__(env_spec, policy)

        if options is None:
            options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 1.e-2, "gpu": True}
        if darkflow_path is None:
            darkflow_path = os.path.join(str(Path.home()), 'darkflow')
        old_cwd = os.getcwd()
        os.chdir(darkflow_path)
        self._tfnet = TFNet(options)
        os.chdir(old_cwd)
        self._labels = []
        for key in self._env_spec.goal_spec:
            if key[-5:] == '_diff':
                self._labels.append(key[:-5])
        self._im_is_rgb = im_is_rgb

    def label(self, observations, curr_goals):
        obs_ims = observations[0]
        goals = []
        for obs_im, curr_goal in zip(obs_ims, curr_goals):
            if self._im_is_rgb:
                obs_im = obs_im[..., ::-1] # make bgr
            im_height, im_width, _ = obs_im.shape

            results = self._tfnet.return_predict(obs_im)
            boxes = self._get_boxes(results)
            goal = self._get_goal(boxes, curr_goal, im_height, im_width)
            goals.append(goal)
        return goals

    def _get_boxes(self, results):
        boxes = []
        for result in results:
            if result['label'] in self._labels:
                boxes.append(result)
        return boxes

    def _get_goal(self, boxes, curr_goal, im_height, im_width):
        # goal is weight of each goal and then desired center pixel
        goals = {}
        min_areas = {}
        for label in self._labels:
            goals[label] = (0., 0.) # (in image, normalized diff to middle)
            min_areas[label] = np.inf

        im_mid = 0.5 * (im_width - 1.)

        for box in boxes:
            label = box['label']
            br_x = box['bottomright']['x']
            br_y = box['bottomright']['y']
            tl_x = box['topleft']['x']
            tl_y = box['topleft']['y']
            mid_point = 0.5 * np.array((br_x + tl_x, br_y + tl_y))
            assert(mid_point[0] >= 0 and mid_point[0] < im_width)
            assert(mid_point[1] >= 0 and mid_point[1] < im_height)
            area = (tl_x - br_x) * (tl_y - br_y)
            if area < min_areas[label]:
                min_areas[label] = area
                norm_diff = (im_mid - mid_point[0]) / im_mid
                goals[label] = (1., norm_diff)  

        goal_keys = list(self._env_spec.goal_spec.keys())
        goal = np.array(curr_goal)

        for label in self._labels:
            idx = min([idx for idx, k in enumerate(goal_keys) if label in k])
            goal[idx:idx+2] = goals[label] # NOTE: assumes in image, normalized diff to middle

        return goal

