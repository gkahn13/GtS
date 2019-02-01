#!/usr/bin/env python

import rospy

import sys

sys.path.append( "/home/crazyflie2/catkin_ws/src/crazyflie/src")

from Crazyflie import Crazyflie

import logging
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

import time
import signal

DEFAULT_URI = 'radio://0/80/250K'

if __name__ == '__main__':

    rospy.init_node('Crazyflie', anonymous=True)

    id_param = rospy.search_param('id')
    uri_param = rospy.search_param('uri')
    data_only_param = rospy.search_param('data_only')
    if not id_param or not uri_param: 
        print("No ID or URI Specified! Abort.")
        sys.exit(0)

    ID = int(rospy.get_param(id_param, '0'))
    URI = rospy.get_param(uri_param, DEFAULT_URI)
    data_only = bool(rospy.get_param(data_only_param, DEFAULT_URI))

 #    if not rospy.has_param('id') or not rospy.has_param('uri'):
 #        print("No ID or URI Specified! Abort.")
	# sys.exit(0)

 #    ID = int(rospy.get_param('id', '0'))
 #    URI = rospy.get_param('uri', DEFAULT_URI)

    cf = Crazyflie(ID, URI, data_only)
    cf.run()
