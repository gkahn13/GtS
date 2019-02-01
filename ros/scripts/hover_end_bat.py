import logging
import time
import signal
import sys

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

def signal_handler(signal, frame):
    print("**** Crazyflie Stopping ****")
    #land seq
    for y in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, (10-y) / 25)
        time.sleep(0.1)
    
    close_terminate();
    
    sys.exit(0)

def close_terminate():
    try:
        cf.commander.send_stop_setpoint()
        log_bat.stop()
        logFile.close()
    except Exception as e:
        print("Error in stopping log: %s" % str(e))


signal.signal(signal.SIGINT, signal_handler)

URI = 'radio://0/80/250K'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

#assume full to start
vbat = 4.0
#somewhat arbitrary
V_THRESH = 3.13

def received_bat_data(timestamp, data, logconf):
    global vbat
    #print('[%d][%s]: %f' % (timestamp, logconf.name, float(data['pm.vbat'])))
    vbat = float(data['pm.vbat'])

def error_bat_data(logconf, msg):
    print('Error when logging %s: %s' % (logconf.name, msg))

if __name__ == '__main__':

    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        cf = scf.cf

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(1.5)

        log_bat = LogConfig(name='Battery', period_in_ms=100)
        log_bat.add_variable('pm.vbat', 'float')

        logFile = open("bat.txt","w+")

        try:
            cf.log.add_config(log_bat)
            # This callback will receive the data
            log_bat.data_received_cb.add_callback(received_bat_data)
            # This callback will be called on errors
            log_bat.error_cb.add_callback(error_bat_data)
            # Start the logging
            log_bat.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Battery log config, bad configuration.')


        print("--------- Initiating Sequence ---------")

        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
            time.sleep(0.1)

        print("**** Crazyflie in the air! ****")
        while vbat > V_THRESH:
            cf.commander.send_hover_setpoint(0, 0, 0, 10 / 25)
            logFile.write(str(vbat) + "\r\n") # write battery voltage to file
            time.sleep(0.05)

        print("**** Low battery detected -- Landing ****")
        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, (10-y) / 25)
            time.sleep(0.1)

        #ENDING
        close_terminate()



