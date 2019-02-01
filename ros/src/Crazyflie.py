import rospy
import numpy as np

from crazyflie.msg import CFData
# from crazyflie.msg import CFImage
# from sensor_msgs.msg import Image
from crazyflie.msg import CFCommand
from crazyflie.msg import CFMotion

import logging
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

import time
import signal
import sys
import os

import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie as CF
from cflib.crazyflie.log import LogConfig

MAX_ALT = 1



## for convenience
cmd_type = ['']*3
cmd_type[CFCommand.ESTOP] = 'ESTOP'
cmd_type[CFCommand.LAND] = 'LAND'
cmd_type[CFCommand.TAKEOFF] = 'TAKEOFF'

# Handles all interaction with CF through its radio.
class Crazyflie:

    # ID is for human readability
    def __init__(self, cf_id, radio_uri, data_only=False):
        self._id = cf_id
        self._uri = radio_uri

        self.stop_sig = False

        signal.signal(signal.SIGINT, self.signal_handler)

        self.cf_active = False

        self.accept_commands = False
        self.data_only = data_only

        self.data = None
        self.alt = 0

        # self.bridge = CvBridge()

        cflib.crtp.init_drivers(enable_debug_driver=False)
        # try:
        # with SyncCrazyflie(self._uri) as scf:

        self.cf = CF(rw_cache="./cache")
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)
        self.cf.connection_failed.add_callback(self.connection_lost)
        self.cf.connection_lost.add_callback(self.connection_failed)

        print('Connecting to %s' % radio_uri)
        self.cf.open_link(radio_uri)

        # self.cf.param.set_value('kalman.resetEstimation', '1')
        # time.sleep(0.1)
        # self.cf.param.set_value('kalman.resetEstimation', '0')
        # time.sleep(1.5)


        # except Exception as e:
        #     print(type(e))
        #     print("Unable to connect to CF %d at URI %s" % (self._id, self._uri))
        #     self.scf = None
        #     self.cf = None

        

        self.data_pub = rospy.Publisher('cf/%d/data'%self._id, CFData, queue_size=10)
        # self.image_pub = rospy.Publisher('cf/%d/image'%self._id, Image, queue_size=10)
        if not self.data_only:
            self.cmd_sub = rospy.Subscriber('cf/%d/command'%self._id, CFCommand, self.command_cb)
            self.motion_sub = rospy.Subscriber('cf/%d/motion'%self._id, CFMotion, self.motion_cb)

    def signal_handler(self, sig, frame):
        if self.cf_active:
            self.cmd_estop()
        self.stop_sig = True
        rospy.signal_shutdown("CtrlC")

        #killing
        os.kill(os.getpgrp(), signal.SIGKILL)

    ## CALLBACKS ##
    def connected(self, uri):
        print("Connected to Crazyflie at URI: %s" % uri)

        self.cf_active = True

        try:
            self.log_data = LogConfig(name="Data", period_in_ms=10)
            self.log_data.add_variable('acc.x', 'float')
            self.log_data.add_variable('acc.y', 'float')
            self.log_data.add_variable('acc.z', 'float')
            self.log_data.add_variable('pm.vbat', 'float')
            self.log_data.add_variable('stateEstimate.z', 'float')
            self.cf.log.add_config(self.log_data)
            self.log_data.data_received_cb.add_callback(self.received_data)


            #begins logging and publishing
            self.log_data.start()

            print("Logging Setup Complete. Starting...")
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add log config, bad configuration.')


    def disconnected(self, uri):
        self.cf_active = False
        print("Disconnected from Crazyflie at URI: %s" % uri)

    def connection_failed(self, uri, msg):
        self.cf_active = False
        print("Connection Failed")

    def connection_lost(self, uri, msg):
        self.cf_active = False
        print("Connection Lost")

    def command_cb(self, msg):
        print("ALT: %.3f" % self.alt)
        if self.accept_commands:
            print("RECEIVED COMMAND : %s" % cmd_type[msg.cmd])
            if cmd_type[msg.cmd] == 'ESTOP':
                self.cmd_estop()
            elif cmd_type[msg.cmd] == 'LAND':
                self.alt = 0
                self.cmd_land()
            elif cmd_type[msg.cmd] == 'TAKEOFF':
                self.alt = 0.4
                self.cmd_takeoff()
            else:
                print('Invalid Command! %d' % msg.cmd)
        elif cmd_type[msg.cmd] == 'TAKEOFF':
            self.alt = 0.4
            self.cmd_takeoff()
        else:
            print("Not Accepting Commands -- but one was sent!")

    def motion_cb(self, msg):
        print("ALT: %.3f" % self.alt)
        print(msg)
        if self.accept_commands:
            self.update_alt(msg)
            # switching between optical flow and roll pitch motion
            if msg.is_flow_motion:
                self.set_flow_motion(msg.x, msg.y, msg.yaw, self.alt)
            else:
                self.set_rp_motion(msg.x, msg.y, msg.yaw, self.alt)

        else:
            print("Not Accepting Motion Commands -- but one was sent!")

    def update_alt(self, msg):
        
        #what exactly does this do?
        #motion.alt = self.data.alt * 100 if self.data.alt > ALT_TOLERANCE else 0
        self.alt += msg.dz
        if self.alt < 0:
            self.alt = 0
        elif self.alt > MAX_ALT:
            self.alt = MAX_ALT

    def received_data(self, timestamp, data, logconf):
        # print("DATA RECEIVED")
        # print(self.data)
        self.data = data
        d = CFData()
        d.ID = self._id
        d.accel_x = float(data['acc.x'])
        d.accel_y = float(data['acc.y'])
        d.accel_z = float(data['acc.z'])
        d.v_batt = float(data['pm.vbat'])
        d.alt = float(data['stateEstimate.z'])
        # d.alt = float(data['posEstimatorAlt.estimatedZ'])
        # print(d.v_batt)
        self.data_pub.publish(d)

    ## COMMANDS ##

    def set_flow_motion(self, vx, vy, yaw, alt):
        self.cf.commander.send_hover_setpoint(vx, vy, yaw, alt)

    def set_rp_motion(self, roll_a, pitch_a, yaw_r, alt):
        self.cf.commander.send_zdistance_setpoint(roll_a, pitch_a, yaw_r, alt)

    def cmd_estop(self):
        print("---- Crazyflie %d Emergency Stopping ----" % self._id)
        self.cf.commander.send_stop_setpoint()
        self.accept_commands = False

    def cmd_takeoff(self, alt=0.4):
        for y in range(10):
            print("taking off")
            self.cf.commander.send_hover_setpoint(0, 0, 0, y / 10 * alt)
            time.sleep(0.1)
        self.accept_commands = True

    def cmd_land(self, alt=0.4):
        if self.accept_commands==False:
            print("cannot land right now")
        else:
            for y in range(10):
                self.cf.commander.send_hover_setpoint(0, 0, 0, alt - (y / 10 * alt))
                time.sleep(0.1)
            self.cmd_estop()

    def run(self):
        print("WAITING FOR ACTIVE CONNECTION")
        while not self.cf_active:
            pass
        print("FOUND ACTIVE CONNECTION")

        #handles image reads
        # threading.Thread(target=self.image_thread).start()

        rate = rospy.Rate(25)

        rospy.spin()

        self.log_data.stop()
