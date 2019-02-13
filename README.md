## Generalization through Simulation: Integrating Simulated and Real Data into Deep Reinforcement Learning for Vision-Based Autonomous Flight

[arXiv paper link](https://arxiv.org/abs/1902.03701)

Click below to view the video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Rb2a6lSQSas/0.jpg)](https://www.youtube.com/watch?v=Rb2a6lSQSas)

### Crazyflie Setup

Our quadrotor setup consists of a Crazyflie modified with an onboard camera. See [here for the list of parts](https://docs.google.com/spreadsheets/d/1g7PMl0FwBluRlBCJa5EDDool2KR_z-pgxrk7phjuJNs/edit?usp=sharing) and [here for for instructions on building the Crazyflie and running the software](https://docs.google.com/document/d/10if8hPpw4YIm9bwp7m7-JSx7UDVShbLpTZVWTOdTYHM/edit?usp=sharing). The ROS code is in the [ros directory](https://github.com/gkahn13/GtS/tree/master/ros) included in this repository, and is a standalone package.

### Software Setup

We run our code using docker.

Build and start the docker image:
```bash
$ cd docker
$ ./gcg-docker.sh build Dockerfile-gibson
$ ./gcg-docker.sh start
```

Your main interface to the docker will be through ssh:
```bash
$ ./gcg-docker.sh ssh
```

If you wish to change the docker image, you must run stop, build and start again:
```bash
$ ./gcg-docker.sh stop
$ ./gcg-docker.sh build Dockerfile-gibson
$ ./gcg-docker.sh start
```

### Running our experiments

Download our [data and models](https://drive.google.com/open?id=1QqgbAZVnjGuynIkuDY3f6gOS3y0LnssA) and place to \<path to GtS\>/data

The relevant experiment files are in the \<path to GtS\>/configs.

To evaluate our method in simulation:
```bash
$ cd scripts
$ python run_gcg_eval.py eval_in_sim -itr 800
```

To train from scratch in simulation:
```bash
$ python run_gcg.py train_in_sim
```

To train on the simulation data we gathered, enter '\<path to GtS\>/data/tfrecords' to the 'offpolicy' parameter in configs/train_tf_records.py and then run:
```bash
$ python run_gcg_train.py train_tf_records
```

To evaluate our pre-trained GtS model in the real world:
```bash
$ python run_gcg_eval.py eval_in_rw -itr 6
```
Note: you may need to run this on your local machine (not docker) because of ROS. See the Dockerfile-gibson for relevant system and python dependencies.

### References

Katie Kang*, Suneel Belkhale*, Gregory Kahn*, Pieter Abbeel, Sergey Levine. "Generalization through Simulation: Integrating Simulated and Real Data into Deep Reinforcement Learning for Vision-Based Autonomous Flight." ICRA 2019

