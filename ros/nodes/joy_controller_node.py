#!/usr/bin/env python

import rospy
import sys

sys.path.append("/home/crazyflie2/catkin_ws/src/crazyflie/src")

from JoyController import JoyController


if __name__ == '__main__':

    rospy.init_node('JoyController', anonymous=True)

    idParam = rospy.search_param('id')
    joyTopicParam = rospy.search_param('joy_topic')
    flowParam = rospy.search_param('is_flow_motion')
    useJoyParam = rospy.search_param('use_joy')
    if not idParam or not joyTopicParam:
        print("No ID or JOY TOPIC Specified! Abort.")
        sys.exit(0)

    ID = int(rospy.get_param(idParam, '0'))
    joy_topic = rospy.get_param(joyTopicParam, '/joy')
    use_joy = rospy.get_param(useJoyParam, 'True')
    is_flow = bool(rospy.get_param(flowParam, 'False'))

    # if not rospy.has_param('id'):
    #     print("No ID or URI Specified! Abort.")
    # sys.exit(0)

    # ID = int(rospy.get_param('id', '0'))


    
    control = JoyController(ID, use_joy, joy_topic, is_flow)
    control.run()
