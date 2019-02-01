import logging
import time
import signal
import sys
import pygame
import sys

import threading

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

URI = 'radio://0/80/250K'

def signal_handler(signal, frame):
    print("**** Crazyflie Stopping ****")
    #land seq
    for y in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, (10-y) / 25)
        time.sleep(0.1)
    global done
    done = True
    close_terminate();
    
    sys.exit(0)

def close_terminate():
    try:
        cf.commander.send_stop_setpoint()
    except Exception as e:
        print("Error in stopping: %s" % str(e))

signal.signal(signal.SIGINT, signal_handler)

XY_VEL = 0.25
YAW_RATE = 40
Z_STEP = 0.02



# -1 = first, 0 = neither, 1 = second
upDown = 0
leftRight = 0
forwardBack = 0
yawLeftRight = 0

done = True

active = []

WIDTH = 200
HEIGHT = 100

# runs key checking
def on_key_down(key):
    global upDown
    global leftRight
    global forwardBack
    global yawLeftRight

    if key in active:
        return

    # print("PRESSED " + str(key))
    active.append(key) 
    if key == key.W:
        # print('Alt Up')
        upDown = -1
    elif key == key.S:
        # print('Alt Down')
        upDown = 1
    elif key == key.A:
        # print('Yaw Left')
        yawLeftRight = -1
    elif key == key.D:
        # print('Yaw Right')
        yawLeftRight = 1
    elif key == key.UP:
        # print('Pitch Forward')
        forwardBack = -1
    elif key == key.DOWN:
        # print('Pitch Back')
        forwardBack = 1
    elif key == key.LEFT:
        # print('Roll Left')
        leftRight = -1
    elif key == key.RIGHT:
        # print('Roll Right')
        leftRight = 1

def on_key_up(key):
    global upDown
    global leftRight
    global forwardBack
    global yawLeftRight

    if key in active:
        active.remove(key)
    # print("RELEASED " + str(key))
    #releasing
    if key == key.W or key == key.S:
        # print('Release Alt')
        upDown = 0
    elif key == key.A or key == key.D:
        # print('Release Yaw')
        yawLeftRight = 0
    elif key == key.UP or key == key.DOWN:
        # print('Release Pitch')
        forwardBack = 0
    elif key == key.LEFT or key == key.RIGHT:
        # print('Release Roll')
        leftRight = 0

cf = None


def main_thread():
    ALT = 0.4
    VX = 0
    VY = 0
    YAW = 0

    global cf

    print("STARTING MAIN")
   
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        cf = scf.cf

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(1.5)

        print("--------- Initiating Sequence ---------")

        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, y * ALT / 10)
            time.sleep(0.1)

        print("**** Crazyflie in the air! ****")
        global done
        done = False

        while not done:
            #ROLL BLOCK
            if leftRight == -1:
                #left
                VY = XY_VEL
            elif leftRight == 1:
                #right
                VY = -XY_VEL
            else:
                VY = 0

            #FORWARD BLOCK
            if forwardBack == -1:
                #forward
                VX = XY_VEL
            elif forwardBack == 1:
                #back
                VX = -XY_VEL
            else:
                VX = 0

            #ALT BLOCK
            if upDown == -1:
                #up
                ALT += Z_STEP
            elif upDown == 1:
                #down
                ALT -= Z_STEP
                if ALT < 0.4:
                    ALT = 0.4

            #YAW BLOCK
            if yawLeftRight == -1:
                #left
                YAW = -YAW_RATE
            elif yawLeftRight == 1:
                #right
                YAW = YAW_RATE
            else:
                YAW = 0
            
            cf.commander.send_hover_setpoint(VX, VY, YAW, ALT)
            print("VALS: (VX %.2f, VY %.2f, YAW %.2f, ALT %.2f" % (VX, VY, YAW, ALT))
            time.sleep(0.1)

        # for y in range(10):
        #     cf.commander.send_hover_setpoint(0, 0, 0, (10-y) / 25)
        #     time.sleep(0.1)

        #ENDING
        #close_terminate()


reader = threading.Thread(target=main_thread)
reader.start()
# done = False
print("STARTING")

# pygame.init()

# pygame.display.set_mode((200, 100))
# while not done:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit()
#         # setting 4 axis
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_w:
#                 print('Alt Up')
#                 upDown = -1
#             elif event.key == pygame.K_s:
#                 print('Alt Down')
#                 upDown = 1
#             elif event.key == pygame.K_a:
#                 print('Yaw Left')
#                 yawLeftRight = -1
#             elif event.key == pygame.K_d:
#                 print('Yaw Right')
#                 yawLeftRight = 1
#             elif event.key == pygame.K_UP:
#                 print('Pitch Forward')
#                 forwardBack = -1
#             elif event.key == pygame.K_DOWN:
#                 print('Pitch Back')
#                 forwardBack = -1
#             elif event.key == pygame.K_LEFT:
#                 print('Roll Left')
#                 leftRight = -1
#             elif event.key == pygame.K_RIGHT:
#                 print('Roll Right')
#                 leftRight = 1
#         #releasing
#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_w or event.key == pygame.K_s:
#                 print('Release Alt')
#                 upDown = 0
#             elif event.key == pygame.K_a or event.key == pygame.K_d:
#                 print('Release Yaw')
#                 yawLeftRight = 0
#             elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
#                 print('Release Pitch')
#                 forwardBack = 0
#             elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
#                 print('Release Roll')
#                 leftRight = 0

# #wait til crazy flie is in the air
# while done:
#     s = 0

# main = Tk()
# frame = Frame(main, width=200, height=100)
# main.bind_all('<KeyPress>', key_press)
# main.bind_all('<KeyRelease>', key_release)
# frame.pack()
# main.mainloop()
# done = True

# # gui.join()
# print("Direction Reader Closed")
# def signal_handler(signal, frame):
#     print("**** Crazyflie Stopping ****")
#     #land seq
#     for y in range(10):
#         cf.commander.send_hover_setpoint(0, 0, 0, (10-y) / 25)
#         time.sleep(0.1)
    
#     close_terminate();
    
#     sys.exit(0)

# def close_terminate():
#     try:
#         cf.commander.send_stop_setpoint()
#         log_bat.stop()
#         logFile.close()
#     except Exception as e:
#         print("Error in stopping log: %s" % str(e))


# signal.signal(signal.SIGINT, signal_handler)

# URI = 'radio://0/80/250K'

# # Only output errors from the logging framework
# logging.basicConfig(level=logging.ERROR)

# #assume full to start
# vbat = 4.0
# #somewhat arbitrary
# V_THRESH = 3.13

# def received_bat_data(timestamp, data, logconf):
#     global vbat
#     #print('[%d][%s]: %f' % (timestamp, logconf.name, float(data['pm.vbat'])))
#     vbat = float(data['pm.vbat'])

# def error_bat_data(logconf, msg):
#     print('Error when logging %s: %s' % (logconf.name, msg))

# if __name__ == '__main__':

#     # Initialize the low-level drivers (don't list the debug drivers)
#     cflib.crtp.init_drivers(enable_debug_driver=False)

#     with SyncCrazyflie(URI) as scf:
#         cf = scf.cf

#         cf.param.set_value('kalman.resetEstimation', '1')
#         time.sleep(0.1)
#         cf.param.set_value('kalman.resetEstimation', '0')
#         time.sleep(1.5)

#         log_bat = LogConfig(name='Battery', period_in_ms=100)
#         log_bat.add_variable('pm.vbat', 'float')

#         logFile = open("bat.txt","w+")

#         try:
#             cf.log.add_config(log_bat)
#             # This callback will receive the data
#             log_bat.data_received_cb.add_callback(received_bat_data)
#             # This callback will be called on errors
#             log_bat.error_cb.add_callback(error_bat_data)
#             # Start the logging
#             log_bat.start()
#         except KeyError as e:
#             print('Could not start log configuration,'
#                   '{} not found in TOC'.format(str(e)))
#         except AttributeError:
#             print('Could not add Battery log config, bad configuration.')


#         print("--------- Initiating Sequence ---------")

#         for y in range(10):
#             cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
#             time.sleep(0.1)

#         print("**** Crazyflie in the air! ****")
#         while vbat > V_THRESH:
#             cf.commander.send_hover_setpoint(0, 0, 0, 10 / 25)
#             logFile.write(str(vbat) + "\r\n") # write battery voltage to file
#             time.sleep(0.05)

#         print("**** Low battery detected -- Landing ****")
#         for y in range(10):
#             cf.commander.send_hover_setpoint(0, 0, 0, (10-y) / 25)
#             time.sleep(0.1)

#         #ENDING
#         close_terminate()



