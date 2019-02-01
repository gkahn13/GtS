#!/usr/bin/python2
import argparse

from PIL import Image

import rospy

import cv_bridge
from cv_bridge import  CvBridge

parser = argparse.ArgumentParser()
parser.add_argument('--impath', type=str, nargs=1, dest="impath", help='path to image')
# parser.add_argument('--topic', type=str, nargs=1, dest="topic", help='path to image')

parser.add_argument('--publish', dest='pub', action='store_true', 
    default=False, help='publish ros image from the path')
# parser.add_argument('--subscribe', dest='sub', action='store_true', 
#     default=False, help='read from ros topic and save to this path')

args = parser.parse_args()
#RUN THIS SCRIPT IN PYTHON 2


if __name__ == '__main__':

    rospy.init_node("temp_bridge")

    bridge = CvBridge()

    if not args.impath:
        print "Must specify --impath!"
        # return
    elif args.pub:
        im = Image.open(args.impath)
        mat = np.array(im)

        # return bridge.cv2_to_imgmsg(mat, mat.dtype.type)

