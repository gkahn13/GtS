#!/usr/bin/python2

import rospy

import cv_bridge
from cv_bridge import CvBridge
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage
from crazyflie.msg import CFData
# from crazyflie.msg import CFImage
from crazyflie.msg import CFCommand
from crazyflie.msg import CFMotion
import time
import matplotlib.pyplot as plt
import os


class Camera:

    # DO_NOTHING_CMD = CFMotion()

    def __init__(self, ID):
        self.id = ID

        self.bridge = CvBridge()

        self.mat = None

        #need to facilitate a set of publishers per cf node
        self.image_pub = rospy.Publisher('cf/%d/image'%self.id, CompressedImage, queue_size=10)


    ## CALLBACKS ## 


    ## THREADS ##
    def run(self):
        try: 
            cap = cv2.VideoCapture(0) # TODO: multiple vid captures in parallel
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 192)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
            # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.8)
            # cap.set(cv2.CAP_PROP_CONTRAST, 0.2)
            # cap.set(cv2.CAP_PROP_EXPOSURE, 0.08)
            # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            while not rospy.is_shutdown():
                
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #ret, gray = cap.read()
                self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(gray))

                cv2.imshow('frame', gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print "CAMERA %d STREAM FAILED -- CHECK INPUTS" % self.id
            print "Error: " + str(e)

        print " -- Camera %d Finished -- " % self.id
