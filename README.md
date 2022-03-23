# gaze_trajectory_Pupil_Xsens
Matlab and Python code allowing to determine the gaze trajectory projevcted on the walls of the gymnasium on trampoline datatc ollected with Xsens IMUs and Pupil eye-tracker. This code also detects anticipatory/compensatory/movement-detection head and eye movements.


# To install:
conda create --name [name] python=3.9

conda activate [name]

conda install -c conda-forge scipy numpy pickle

Also need matlab

# Workflow:
1. Exports Xsens data through MVN analyse (make sure all data export options are set to true) -> saves IMU data to [trial_name].mvnx
2. Modify the .mvnx file to make it readable with main gaze_mapping.m -> saves IMU data to [trial_name].mat
3. Synchornize the data from IMU and eye-tracker with synchronize.py -> saves time stamps ...



