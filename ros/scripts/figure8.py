import logging
import time
import signal
import sys

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

float xy_vel = 0.

def signal_handler(signal, frame):
    print("**** Crazyflie Stopping ****")

    cf.commander.send_stop_setpoint()

signal.signal(signal.SIGINT, signal_handler)

URI = 'radio://0/80/250K'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':

    print("--------- Initiating Sequence ---------")

    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        cf = scf.cf

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
            time.sleep(0.1)

        print("**** Crazyflie in the air! ****")

        while(True):

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
            time.sleep(0.1)

        for _ in range(50):
            cf.commander.send_hover_setpoint(0.5, 0, 36 * 2, 0.4)
            time.sleep(0.1)

        for _ in range(50):
            cf.commander.send_hover_setpoint(0.5, 0, -36 * 2, 0.4)
            time.sleep(0.1)

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
            time.sleep(0.1)

        print("**** Crazyflie Landing ****")

        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
            time.sleep(0.1)

        cf.commander.send_stop_setpoint()