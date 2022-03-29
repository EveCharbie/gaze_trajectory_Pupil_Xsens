
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import sympy
import scipy.io as sio

file_dir = '/home/user/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData/'
file_name = 'Test_17032021-003/'
Xsens_Subject_name = sio.loadmat(file_dir + file_name + 'Subject_name.mat')["Subject_name"]
Xsens_Move_name = sio.loadmat(file_dir + file_name + 'Move_name.mat')["Move_name"]
Xsens_Subject_name = sio.loadmat(file_dir + file_name + 'Subject_name.mat')["Subject_name"]
Xsens_frameRate = sio.loadmat(file_dir + file_name + 'frameRate.mat')["frameRate"]
Xsens_time = sio.loadmat(file_dir + file_name + 'time.mat')["time"]
Xsens_index = sio.loadmat(file_dir + file_name + 'index.mat')["index"]
Xsens_ms = sio.loadmat(file_dir + file_name + 'ms.mat')["ms"]

Xsens_orientation = sio.loadmat(file_dir + file_name + 'orientation.mat')
Xsens_velocity = sio.loadmat(file_dir + file_name + 'velocity.mat')["velocity"]
Xsens_acceleration = sio.loadmat(file_dir + file_name + 'acceleration.mat')["acceleration"]
Xsens_angularVelocity = sio.loadmat(file_dir + file_name + 'angularVelocity.mat')["angularVelocity"]
Xsens_angularAcceleration = sio.loadmat(file_dir + file_name + 'angularAcceleration.mat')["angularAcceleration"]
Xsens_sensorFreeAcceleration = sio.loadmat(file_dir + file_name + 'sensorFreeAcceleration.mat')["sensorFreeAcceleration"]
Xsens_vsensorOrientation = sio.loadmat(file_dir + file_name + 'vsensorOrientation.mat')["vsensorOrientation"]
Xsens_jointAngle = sio.loadmat(file_dir + file_name + 'jointAngle.mat')["jointAngle"]
Xsens_centerOfMass = sio.loadmat(file_dir + file_name + 'centerOfMass.mat')["centerOfMass"]




















