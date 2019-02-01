import rospy

from sensor_msgs.msg import Image
from crazyflie.msg import CFData
# from crazyflie.msg import CFImage
from crazyflie.msg import CFCommand
from crazyflie.msg import CFMotion

from sensor_msgs.msg import Joy

from Controller import Controller

import signal

THROTTLE_AXIS = 5 # up 1
ROLL_AXIS = 2 #left 1
PITCH_AXIS = 3 #up 1
YAW_AXIS = 0 #left 1

#RP motion
THROTTLE_SCALE = 0.1
ROLL_SCALE = 0.5
PITCH_SCALE = 0.5
YAW_SCALE = -120

#standard motion
VX_SCALE = 0.5
VY_SCALE = 0.5

TAKEOFF_CHANNEL = 7 #RT
ESTOP_CHANNEL = 2 #B
LAND_CHANNEL = 6 #LT
UNLOCK_ESTOP_CHANNEL = 0 #X

TOLERANCE = 0.05
ALT_TOLERANCE = 0.08


#A is 1, Y is 3

class JoyController(Controller):

    def __init__(self, ID, use_joy, joystick_topic, flow_motion=True):
    
        Controller.__init__(self, ID)
        self.use_joy = use_joy

        if self.use_joy:
            self.joy_sub = rospy.Subscriber(joystick_topic, Joy, self.joy_cb)
        else:
            print("------ JOYSTICK NOT BEING USED BY CONTROLLER NODE ------")
        self.curr_joy = None

        self.cmd = -1 # -1 : NONE

        self.is_flow_motion = True#flow_motion

    #Override
    def compute_motion(self):
        #pulls latest joystick data
        if not self.use_joy:
            # no motion input from controller
            return None

        motion = None

        if self.cmd != -1:
            motion = CFCommand()
            if self.cmd == CFCommand.ESTOP:
                motion.cmd = CFCommand.ESTOP

            elif self.cmd == CFCommand.TAKEOFF:
                motion.cmd = CFCommand.TAKEOFF

            elif self.cmd == CFCommand.LAND:
                motion.cmd = CFCommand.LAND

            #reset
            self.cmd = -1

        #repeat send at 10Hz
        elif self.curr_joy:
            motion = CFMotion()

            motion.is_flow_motion = self.is_flow_motion
                # computing regular vx, vy, yaw, alt motion

            if self.is_flow_motion:
                motion.y = self.curr_joy.axes[ROLL_AXIS] * VY_SCALE
                motion.x = self.curr_joy.axes[PITCH_AXIS] * VX_SCALE
            else:
                motion.y = self.curr_joy.axes[ROLL_AXIS] * ROLL_SCALE
                motion.x = self.curr_joy.axes[PITCH_AXIS] * PITCH_SCALE

            #common
            motion.yaw = self.curr_joy.axes[YAW_AXIS] * YAW_SCALE

                
            # print(self.curr_joy.axes)
            motion.dz = self.curr_joy.axes[THROTTLE_AXIS] * THROTTLE_SCALE
            # print("ALT CHANGE: %.3f" % motion.dz)
            
        return motion
    


    def dead_band(self, signal):
        new_axes = [0] * len(signal.axes)
        for i in range(len(signal.axes)):
            new_axes[i] = signal.axes[i] if abs(signal.axes[i]) > TOLERANCE else 0
        signal.axes = new_axes


    def joy_cb(self, msg):
        if self.curr_joy:
            if msg.buttons[ESTOP_CHANNEL] and not self.curr_joy.buttons[ESTOP_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.ESTOP
                print("CALLING ESTOP")
            elif msg.buttons[TAKEOFF_CHANNEL] and not self.curr_joy.buttons[TAKEOFF_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.TAKEOFF
                print("CALLING TAKEOFF")
            elif msg.buttons[LAND_CHANNEL] and not self.curr_joy.buttons[LAND_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.LAND
                print("CALLING LAND")
        else:
            if msg.buttons[ESTOP_CHANNEL] :
                #takeoff
                self.cmd = CFCommand.ESTOP
                print("CALLING ESTOP")
            elif msg.buttons[TAKEOFF_CHANNEL] :
                #takeoff
                self.cmd = CFCommand.TAKEOFF
                print("CALLING TAKEOFF")
            elif msg.buttons[LAND_CHANNEL] :
                #takeoff
                self.cmd = CFCommand.LAND
                print("CALLING LAND")

        self.dead_band(msg)
        self.curr_joy = msg
