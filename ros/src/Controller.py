import rospy
import numpy as np
from PIL import Image

import io

import cv2
import math

from sensor_msgs.msg import CompressedImage
from crazyflie.msg import CFData
# from crazyflie.msg import CFImage
from crazyflie.msg import CFCommand
from crazyflie.msg import CFMotion
from std_msgs.msg import Bool


## for convenience
cmd_type = ['']*3
cmd_type[CFCommand.ESTOP] = 'ESTOP'
cmd_type[CFCommand.LAND] = 'LAND'
cmd_type[CFCommand.TAKEOFF] = 'TAKEOFF'

class Controller:

    DO_NOTHING_CMD = CFMotion()

    COLLISION_THRESH = 0.5


    def __init__(self, ID):
        self.id = ID

        self.mat = None
        self.data = None

        #need to facilitate a set of publishers per cf node

        self.data_sub = rospy.Subscriber('cf/%d/data' % ID, CFData, self.data_cb)
        self.image_sub = rospy.Subscriber('cf/%d/image' % ID, CompressedImage, self.image_cb)

        self.cmd_pub = rospy.Publisher('cf/%d/command'% self.id, CFCommand, queue_size=10)
        self.motion_pub = rospy.Publisher('cf/%d/motion'% self.id, CFMotion, queue_size=10)
        self.coll_pub = rospy.Publisher('cf/%d/coll'% self.id, Bool, queue_size=10)

    def compute_motion(self):
        print("Doing nothing -- ")
        return None

    def observedCollision(self):
        # if not self.data:
        #     return False
        # xy_mag = math.sqrt(self.data.accel_x ** 2 + self.data.accel_y ** 2)
        # print("-- COLLIDED --") if xy_mag > self.COLLISION_THRESH else None
        # return xy_mag > self.COLLISION_THRESH
        return False



    ## CALLBACKS ## 

    def image_cb(self, msg):
        pil_jpg = io.BytesIO(msg.data)
        pil_arr = Image.open(pil_jpg).convert('L') #grayscale

        self.mat = np.array(pil_arr)
        pass

    def data_cb(self, msg):
        self.data = msg
        pass

    ## SETTERS ##

    ## THREADS ##
    def run(self):
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            action = self.compute_motion()

            if self.data and self.observedCollision():
                print("-- COLLISION : E-stopping --")
                action = CFCommand()
                action.cmd = CFCommand.ESTOP
                self.cmd_pub.publish(action)
                self.coll_pub.publish(True)
                #sleep for 2 seconds
                rospy.Rate(1.0/2).sleep()
                print("Back Online")
            else:
                self.coll_pub.publish(False)


            if isinstance(action, CFMotion):
                # if action != Controller.DO_NOTHING_CMD:
                self.motion_pub.publish(action)

                # else:
                #     print("--- DO NOTHING CMD SENT ---")
            elif isinstance(action, CFCommand):
                self.cmd_pub.publish(action)
                print( "CALLED COMMAND -> %s" % cmd_type[action.cmd])

            else:
                pass

            # rospy.spinOnce()
            rate.sleep()
