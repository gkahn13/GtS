#!/usr/bin/env python

import subprocess
import os

import rospy

if __name__ == '__main__':
    rospy.init_node('Camera', anonymous=True)

    id_param = rospy.search_param('id')
    if not id_param: 
        print("No ID Specified! Abort.")
        sys.exit(0)

    ID = int(rospy.get_param(id_param, '0'))

    path = os.path.join(os.getenv('HOME'), "catkin_ws", "src", "crazyflie", "nodes", "py2_cam_node.py")
    py2_cmd = "python2 %s %d" % (path, ID)

    p = subprocess.Popen(py2_cmd.split())

    while not rospy.is_shutdown():
    	pass

    print(" -- Remotely Terminating Camera Node -- ")
    p.kill()
