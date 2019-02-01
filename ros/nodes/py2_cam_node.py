#!/usr/bin/python2

import sys
import rospy

# sys.path.append( "$HOME/catkin_ws/src/crazyflie/src")
import crazyflie
from Camera import Camera

import logging
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

import time
import signal

DEFAULT_URI = 'radio://0/80/250K'

if __name__ == '__main__':

    rospy.init_node('Py2Camera', anonymous=True)

    if len(sys.argv) == 2: 
        ID = sys.argv[1]

        cam = Camera(int(ID))
        cam.run()

    else:
        print "More/Less than 2 arguments passed in"
