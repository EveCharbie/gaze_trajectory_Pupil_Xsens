
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import sympy
import scipy.io as sio
from scipy import signal
from IPython import embed
import pandas as pd
from dtw import *


############################### Load Xsens data ##############################################

segment_Q_label = [ 'Pelvis_w', 'Pelvis_i', 'Pelvis_j', 'Pelvis_k', 'L5_w', 'L5_i', 'L5_j', 'L5_k', 'L3_w', 'L3_i',
                    'L3_j', 'L3_k', 'T12_w', 'T12_i', 'T12_j', 'T12_k', 'T8_w', 'T8_i', 'T8_j', 'T8_k', 'Neck_w',
                    'Neck_i', 'Neck_j', 'Neck_k', 'Head_w', 'Head_i', 'Head_j', 'Head_k', 'RightShoulder_w',
                    'RightShoulder_i', 'RightShoulder_j', 'RightShoulder_k', 'RightUpperArm_w', 'RightUpperArm_i',
                    'RightUpperArm_j', 'RightUpperArm_k', 'RightForeArm_w', 'RightForeArm_i', 'RightForeArm_j',
                    'RightForeArm_k', 'RightHand_w', 'RightHand_i', 'RightHand_j', 'RightHand_k', 'LeftShoulder_w',
                    'LeftShoulder_i', 'LeftShoulder_j', 'LeftShoulder_k', 'LeftUpperArm_w', 'LeftUpperArm_i',
                    'LeftUpperArm_j', 'LeftUpperArm_k', 'LeftForeArm_w', 'LeftForeArm_i', 'LeftForeArm_j',
                    'LeftForeArm_k', 'LeftHand_w', 'LeftHand_i', 'LeftHand_j', 'LeftHand_k', 'RightUpperLeg_w',
                    'RightUpperLeg_i', 'RightUpperLeg_j', 'RightUpperLeg_k', 'RightLowerLeg_w', 'RightLowerLeg_i',
                    'RightLowerLeg_j', 'RightLowerLeg_k', 'RightFoot_w', 'RightFoot_i', 'RightFoot_j', 'RightFoot_k',
                    'RightToe_w', 'RightToe_i', 'RightToe_j', 'RightToe_k', 'LeftUpperLeg_w', 'LeftUpperLeg_i',
                    'LeftUpperLeg_j', 'LeftUpperLeg_k', 'LeftLowerLeg_w', 'LeftLowerLeg_i', 'LeftLowerLeg_j',
                    'LeftLowerLeg_k', 'LeftFoot_w', 'LeftFoot_i', 'LeftFoot_j', 'LeftFoot_k', 'LeftToe_w', 'LeftToe_i',
                    'LeftToe_j', 'LeftToe_k'] # 92

segement_labels = ['Pelvis_x', 'Pelvis_y', 'Pelvis_z', 'L5_x', 'L5_y', 'L5_z', 'L3_x', 'L3_y', 'L3_z', 'T12_x', 'T12_y',
                   'T12_z', 'T8_x', 'T8_y', 'T8_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z',
                   'RightShoulder_x', 'RightShoulder_y', 'RightShoulder_z', 'RightUpperArm_x', 'RightUpperArm_y',
                   'RightUpperArm_z', 'RightForeArm_x', 'RightForeArm_y', 'RightForeArm_z', 'RightHand_x',
                   'RightHand_y','RightHand_z', 'LeftShoulder_x', 'LeftShoulder_y', 'LeftShoulder_z', 'LeftUpperArm_x',
                   'LeftUpperArm_y', 'LeftUpperArm_z', 'LeftForeArm_x', 'LeftForeArm_y', 'LeftForeArm_z', 'LeftHand_x',
                   'LeftHand_y', 'LeftHand_z', 'RightUpperLeg_x', 'RightUpperLeg_y', 'RightUpperLeg_z',
                   'RightLowerLeg_x', 'RightLowerLeg_y', 'RightLowerLeg_z', 'RightFoot_x', 'RightFoot_y', 'RightFoot_z',
                   'RightToe_x', 'RightToe_y', 'RightToe_z', 'LeftUpperLeg_x', 'LeftUpperLeg_y', 'LeftUpperLeg_z',
                   'LeftLowerLeg_x', 'LeftLowerLeg_y', 'LeftLowerLeg_z', 'LeftFoot_x', 'LeftFoot_y', 'LeftFoot_z',
                   'LeftToe_x', 'LeftToe_y', 'LeftToe_z'] # 69

sensors_labels = ['Pelvis_x', 'Pelvis_y', 'Pelvis_z', 'T8_x', 'T8_y', 'T8_z', 'Head_x', 'Head_y', 'Head_z',
                   'RightShoulder_x', 'RightShoulder_y', 'RightShoulder_z', 'RightUpperArm_x', 'RightUpperArm_y',
                   'RightUpperArm_z', 'RightForeArm_x', 'RightForeArm_y', 'RightForeArm_z', 'RightHand_x',
                   'RightHand_y', 'RightHand_z', 'LeftShoulder_x', 'LeftShoulder_y', 'LeftShoulder_z',
                   'LeftUpperArm_x', 'LeftUpperArm_y', 'LeftUpperArm_z', 'LeftForeArm_x', 'LeftForeArm_y',
                   'LeftForeArm_z', 'LeftHand_x', 'LeftHand_y', 'LeftHand_z', 'RightUpperLeg_x', 'RightUpperLeg_y',
                   'RightUpperLeg_z', 'RightLowerLeg_x', 'RightLowerLeg_y', 'RightLowerLeg_z', 'RightFoot_x',
                   'RightFoot_y', 'RightFoot_z', 'LeftUpperLeg_x', 'LeftUpperLeg_y', 'LeftUpperLeg_z',
                   'LeftLowerLeg_x', 'LeftLowerLeg_y', 'LeftLowerLeg_z', 'LeftFoot_x', 'LeftFoot_y', 'LeftFoot_z'] # 51

joint_labels = ['jL5S1_x', 'jL5S1_y', 'jL5S1_z', 'jL4L3_x', 'jL4L3_y', 'jL4L3_z', 'jL1T12_x', 'jL1T12_y', 'jL1T12_z',
                'jT9T8_x', 'jT9T8_y', 'jT9T8_z', 'jT1C7_x', 'jT1C7_y', 'jT1C7_z', 'jC1Head_x', 'jC1Head_y', 'jC1Head_z',
                'jRightT4Shoulder…', 'jRightT4Shoulder…', 'jRightT4Shoulder…', 'jRightShoulder_x', 'jRightShoulder_y',
                'jRightShoulder_z', 'jRightElbow_x', 'jRightElbow_y', 'jRightElbow_z', 'jRightWrist_x', 'jRightWrist_y',
                'jRightWrist_z', 'jLeftT4Shoulder_x', 'jLeftT4Shoulder_y', 'jLeftT4Shoulder_z', 'jLeftShoulder_x',
                'jLeftShoulder_y', 'jLeftShoulder_z', 'jLeftElbow_x', 'jLeftElbow_y', 'jLeftElbow_z', 'jLeftWrist_x',
                'jLeftWrist_y', 'jLeftWrist_z', 'jRightHip_x', 'jRightHip_y', 'jRightHip_z', 'jRightKnee_x',
                'jRightKnee_y', 'jRightKnee_z', 'jRightAnkle_x', 'jRightAnkle_y', 'jRightAnkle_z', 'jRightBallFoot_x',
                'jRightBallFoot_y', 'jRightBallFoot_z', 'jLeftHip_x', 'jLeftHip_y', 'jLeftHip_z', 'jLeftKnee_x',
                'jLeftKnee_y', 'jLeftKnee_z', 'jLeftAnkle_x', 'jLeftAnkle_y', 'jLeftAnkle_z', 'jLeftBallFoot_x',
                'jLeftBallFoot_y', 'jLeftBallFoot_z'] # 66

sensors_Qlabels = ['Pelvis_w', 'Pelvis_i', 'Pelvis_j', 'Pelvis_k', 'T8_w', 'T8_i', 'T8_j', 'T8_k', 'Head_w', 'Head_i',
                   'Head_j', 'Head_k', 'RightShoulder_w', 'RightShoulder_i', 'RightShoulder_j', 'RightShoulder_k',
                   'RightUpperArm_w', 'RightUpperArm_i', 'RightUpperArm_j', 'RightUpperArm_k', 'RightForeArm_w',
                   'RightForeArm_i', 'RightForeArm_j', 'RightForeArm_k', 'RightHand_w', 'RightHand_i', 'RightHand_j',
                   'RightHand_k', 'LeftShoulder_w', 'LeftShoulder_i', 'LeftShoulder_j', 'LeftShoulder_k',
                   'LeftUpperArm_w', 'LeftUpperArm_i', 'LeftUpperArm_j', 'LeftUpperArm_k', 'LeftForeArm_w',
                   'LeftForeArm_i', 'LeftForeArm_j', 'LeftForeArm_k', 'LeftHand_w', 'LeftHand_i', 'LeftHand_j',
                   'LeftHand_k', 'RightUpperLeg_w', 'RightUpperLeg_i', 'RightUpperLeg_j', 'RightUpperLeg_k',
                   'RightLowerLeg_w', 'RightLowerLeg_i', 'RightLowerLeg_j', 'RightLowerLeg_k', 'RightFoot_w',
                   'RightFoot_i', 'RightFoot_j', 'RightFoot_k', 'LeftUpperLeg_w', 'LeftUpperLeg_i', 'LeftUpperLeg_j',
                   'LeftUpperLeg_k', 'LeftLowerLeg_w', 'LeftLowerLeg_i', 'LeftLowerLeg_j', 'LeftLowerLeg_k',
                   'LeftFoot_w', 'LeftFoot_i', 'LeftFoot_j', 'LeftFoot_k'] # 68


file_dir = '/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData/'
file_name = 'Test_17032021-003/'
Xsens_Subject_name = sio.loadmat(file_dir + file_name + 'Subject_name.mat')["Subject_name"]
Xsens_Move_name = sio.loadmat(file_dir + file_name + 'Move_name.mat')["Move_name"]
Xsens_Subject_name = sio.loadmat(file_dir + file_name + 'Subject_name.mat')["Subject_name"]
Xsens_frameRate = sio.loadmat(file_dir + file_name + 'frameRate.mat')["frameRate"]
Xsens_time = sio.loadmat(file_dir + file_name + 'time.mat')["time"]
Xsens_index = sio.loadmat(file_dir + file_name + 'index.mat')["index"]
Xsens_ms = sio.loadmat(file_dir + file_name + 'ms.mat')["ms"]

# Xsens_orientation = sio.loadmat(file_dir + file_name + 'orientation.mat')
Xsens_velocity = sio.loadmat(file_dir + file_name + 'velocity.mat')["velocity"]
# Xsens_acceleration = sio.loadmat(file_dir + file_name + 'acceleration.mat')["acceleration"]
# Xsens_angularVelocity = sio.loadmat(file_dir + file_name + 'angularVelocity.mat')["angularVelocity"]
# Xsens_angularAcceleration = sio.loadmat(file_dir + file_name + 'angularAcceleration.mat')["angularAcceleration"]
Xsens_sensorFreeAcceleration = sio.loadmat(file_dir + file_name + 'sensorFreeAcceleration.mat')["sensorFreeAcceleration"]
# Xsens_sensorOrientation = sio.loadmat(file_dir + file_name + 'sensorOrientation.mat')["sensorOrientation"]
# Xsens_jointAngle = sio.loadmat(file_dir + file_name + 'jointAngle.mat')["jointAngle"]
# Xsens_centerOfMass = sio.loadmat(file_dir + file_name + 'centerOfMass.mat')["centerOfMass"]




############################### Load Pupil data ############################################################

eye_tracking_data_path = '/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/PupilData/2021-08-18_13-12-52-de913cc7/'
filename_gaze = eye_tracking_data_path  + 'gaze.csv'
filename_imu = eye_tracking_data_path  + 'imu.csv'
filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'

csv_gaze_read = np.char.split(pd.read_csv(filename_gaze, sep='\t').values.astype('str'), sep=',')
csv_imu_read = np.char.split(pd.read_csv(filename_imu, sep='\t').values.astype('str'), sep=',')
timestamps_read = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')

time_stamps_eye_tracking = np.zeros((len(timestamps_read), ))
for i in range(len(timestamps_read)):
    time_stamps_eye_tracking[i] = float(timestamps_read[i][0][2])

csv_eye_tracking = np.zeros((len(csv_gaze_read), 7))
for i in range(len(csv_gaze_read)):
    csv_eye_tracking[i, 0] = float(csv_gaze_read[i][0][2]) # timestemp
    csv_eye_tracking[i, 1] = int(round(float(csv_gaze_read[i][0][3]))) # pos_x
    csv_eye_tracking[i, 2] = int(round(float(csv_gaze_read[i][0][4]))) # pos_y
    csv_eye_tracking[i, 3] = float(csv_gaze_read[i][0][5]) # confidence
    csv_eye_tracking[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - time_stamps_eye_tracking)) # closest image timestemp

# 2 -> 0: gaze_timestamp
# 3 -> 1: norm_pos_x
# 4 -> 2: norm_pos_y
# 5 -> 3: confidence
# 4: closest image time_stamp
# 5: pos_x_bedFrame -> computed from labeling and distortion
# 6: pos_y_bedFrame -> computed from labeling and distortion

csv_imu = np.zeros((len(csv_imu_read), 7))
for i in range(len(csv_gaze_read)):
    csv_imu[i, 0] = float(csv_imu_read[i][0][2]) # timestemp
    csv_imu[i, 1] = float(csv_imu_read[i][0][3]) # gyro_x [deg/s]
    csv_imu[i, 2] = float(csv_imu_read[i][0][4])  # gyro_y [deg/s]
    csv_imu[i, 3] = float(csv_imu_read[i][0][5])  # gyro_z [deg/s]
    csv_imu[i, 4] = float(csv_imu_read[i][0][6]) * 9.81  # acceleration_x [était en G, maintenant en m/s**2]
    csv_imu[i, 5] = float(csv_imu_read[i][0][7]) * 9.81  # acceleration_y [était en G, maintenant en m/s**2]
    csv_imu[i, 6] = float(csv_imu_read[i][0][8]) * 9.81  # acceleration_z [était en G, maintenant en m/s**2]
    # csv_imu[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - time_stamps_eye_tracking)) # closest image timestemp

# 2 -> 0: imu_timestamp
# 3 -> 1: gyro_x
# 4 -> 2: gyro_y
# 5 -> 3: gyro_z
# 6 -> 4: acceleration_x
# 7 -> 5: acceleration_y
# 8 -> 6: acceleration_z



sos = signal.butter(10, 5, 'hp', fs=200, output='sos')
Acceleration_filtered_imu = np.zeros((8000, 3))
# plt.figure()
for idx, i in enumerate([4, 5, 6]):
    Acceleration_filtered_imu[:, idx] = signal.sosfilt(sos, csv_imu[:8000, i])
#     plt.plot(Acceleration_filtered[:, i])
# plt.show()

sos = signal.butter(10, 5, 'hp', fs=60, output='sos')
Acceleration_filtered = np.zeros(np.shape(Xsens_sensorFreeAcceleration))
# plt.figure()
for i in [6, 7, 8]:
    Acceleration_filtered[:, i] = signal.sosfilt(sos,  Xsens_sensorFreeAcceleration[:, i])
#     plt.plot(Acceleration_filtered[:, i])
# plt.show()


norm_acceleration_imu = np.linalg.norm(Acceleration_filtered_imu, axis=1)
norm_acceleration_tete = np.linalg.norm(Acceleration_filtered[:, 6: 9], axis=1)

plt.figure()
plt.plot(norm_acceleration_imu, '-r', label='Pupil')
plt.plot(norm_acceleration_tete, '-g', label='Xsens')
plt.legend()
plt.show()




dtw_output = dtw(norm_acceleration_tete, norm_acceleration_imu, keep_internals=True, window_type='sakoechiba', open_end=True, open_begin=True, step_pattern='asymmetric', window_args={window_size=2000})
dtw_output.plot(type="threeway")
dtw_output.plot(type="twoway", offset=-2)

plt.figure()
plt.plot(norm_acceleration_tete)
plt.plot(dtw_output.index1, norm_acceleration_imu[dtw_output.index2])
plt.show()
embed()









# plt.figure()
# for i in [18, 19, 20, 30, 31, 32]:
#     plt.plot(Xsens_velocity[:, i])
# plt.show()
#
sos = signal.butter(10, 5, 'hp', fs=60, output='sos')
# # filtered = signal.sosfilt(sos, sig)
# # ax2.plot(t, filtered)
# # ax2.set_title('After 15 Hz high-pass filter')
# # ax2.axis([0, 1, -2, 2])
# # ax2.set_xlabel('Time [seconds]')
# # plt.tight_layout()
# # plt.show()
#
Acceleration_filtered = np.zeros(np.shape(Xsens_sensorFreeAcceleration))
# plt.figure()
for i in [0, 1, 2, 18, 19, 20, 30, 31, 32]:
    Acceleration_filtered[:, i] = signal.sosfilt(sos, Xsens_sensorFreeAcceleration[:, i])
#     plt.plot(Acceleration_filtered[:, i])
# plt.show()


norm_acceleration_pelvis = np.linalg.norm(Acceleration_filtered[:, 0: 3], axis=1)
norm_acceleration_handr = np.linalg.norm(Acceleration_filtered[:, 18:21], axis=1)
norm_acceleration_handg = np.linalg.norm(Acceleration_filtered[:, 30:33], axis=1)
norm_velocity_pelvis = np.linalg.norm(Xsens_velocity[:, 0: 3], axis=1)
norm_velocity_handr = np.linalg.norm(Xsens_velocity[:, 18: 21], axis=1)
norm_velocity_handg = np.linalg.norm(Xsens_velocity[:, 30: 33], axis=1)

plt.figure()
plt.plot(norm_acceleration_handr, '-r')
plt.plot(norm_acceleration_handg, '-g')
plt.plot(norm_acceleration_pelvis, '-m')
plt.plot(norm_velocity_handr, '-r')
plt.plot(norm_velocity_handg, '-g')
plt.plot(norm_velocity_pelvis, '-m')
plt.show()
















