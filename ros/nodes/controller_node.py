#!/usr/bin/env python

import rospy
import sys

sys.path.append("/home/crazyflie2/catkin_ws/src/crazyflie/src")

from Controller import Controller


if __name__ == '__main__':

    rospy.init_node('Controller', anonymous=True)

    idParam = rospy.search_param('id')
    if not idParam:
        print("No ID or URI Specified! Abort.")
        sys.exit(0)

    ID = int(rospy.get_param(idParam, '0'))

    # if not rospy.has_param('id'):
    #     print("No ID or URI Specified! Abort.")
    # sys.exit(0)

    # ID = int(rospy.get_param('id', '0'))


    
    control = Controller(ID)
    control.run()
