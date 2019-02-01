# CRAZYFLIE ROS PACKAGE

## Overview
--------------
This package allows for a swarm of Crazyflies to be controlled by a Ground Station computer. The basic idea here is that each Crazyflie (separated by ID) can be controlled separately through its own command and motion topics, and the data and images any Crazyflie publishes are all received by a Controller node. Instantiating a Crazyflie node should be done via the launch file cf.launch. You must have parameters 'id' and 'uri' for each launch file call. We are currently looking into ways to use only one Radio URI to control multiple Crazyflies, which looks possible. (see here: https://forum.bitcraze.io/viewtopic.php?t=624).

## Installation
--------------
1. Setup ROS Kinetic, and setup your ROS Workspace with catkin_ws in your ~/ path.
2. Make sure to have OpenCV3 installed for python 2 (ROS works better with OpenCV in python 2).
3. In python 3, you must pip install the cflib classes from this repository: https://github.com/bitcraze/crazyflie-lib-python. If you are on linux, make sure to also follow the "Setting udev permissions" section.
3. Clone this repository to your ~/catkin_ws/src/
4. cd ~/catkin_ws
5. catkin_make
6. source ~/catkin_ws/devel/setup.bash


## Functionality
--------------
###### Crazyflie {id}:
* subscribes to cf/{id}/command and cf/{id}/motion
* publishes to cf/data and cf/image


###### Controller {id}:
* subscribes to cf/{id}/data and cf/{id}/image
* publishes to cf/{id}/command and cf/{id}/motion


## ROS Layout
--------------
#### NODES
* cf_node.py
* controller_node.py


#### TOPICS
* /cf/{id}/images
* /cf/{id}/data
* /cf/{id}/motion
* /cf/{id}/command
  

#### MESSAGES

<!-- ###### CFImage
* uint16 ID
* sensor_msgs/Image fpv_image
 -->
###### CFData
* uint16 ID
* float accel_x
* float accel_y
* float accel_z

###### CFMotion
* float vel_x
* float vel_y
* float altitude

###### CFCommand
* EMERGENCY-STOP = 0, LAND = 1, TAKEOFF = 2
* int cmd


## Launch Files and Scripts
-------------
#### Launch Files
* cf.launch: Launches the cf\_node, the default controller node, and the camera
* joy.launch: Launches the joystick and the joystick controller nodes.
* joycf.launch: This is the main launch file to start the cf\_node, the joystick controller node, the joystick node, and the camera node.

#### Scripts
* figure8.py: Moves CF in a figure 8.
* hover\_end\_bat.py: uses altitude and optical flow position control to hover until the battery dips below a certain voltage.
* key_control.py: keyboard control of crazyflie (forward, back, left, right) example



