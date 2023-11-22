# Path Planning for Autonomous Parking Using A2C RL Method

This project is part of the ME5418 - Machine Learning for Robotics module from the National University of Singapore. The aim of this project is to use reinforcement learning techniques and most precisely A2C method, to determine the most efficient path and parking maneuver for a vehicle to navigate in complex and crowded parking spaces with narrow corridors and static obstacles (such as parking pillars, walls, and parked cars) before parking in an available spot. To simulate real-world scenarios, our robot, with its associated hitbox, is placed in a 2D world represented by an N × N grid with multiple static obstacles that mimic those encountered by a real car in a parking lot, along with two available parking spots.

# Dependencies

The follwoing are the required libraries and dependencies in order to be able to run and use our code:

* python>=3.9
* torch==2.1.0
* numpy>=1.26
* gym==0.26.2
* scikit-image==0.21.0
* matplotlib==3.8.0
* tensorboard==2.15.1

# Setup Linux

To be able to run the code you have to create a virtual environment by typing these lines in your terminal:
```
$ virtualenv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```
# Usage

* To train a new network : run train.py
* To test a preTrained network : run test.py
* All parameters and hyperparamters to control training / testing / images are in paramter.py

# Training Process Visualization (Tensorboard)

In order to visualize the and keep track of the training process, you need to  access the Tensorboard module via the terminal using the following command line:
tensorboard --logdir=./exp
