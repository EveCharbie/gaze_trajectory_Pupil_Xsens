
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.io as sio
from scipy import signal
from IPython import embed
import pandas as pd
import biorbd
import bioviz


model_path = "/home/user/Documents/Programmation/gaze_trajectory_Pupil_Xsens/models/GuSe_model.bioMod"

file_dir = '/home/user/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData/'
# file_dir = '/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData/'
file_name = 'Test_17032021-003/'

# eye_tracking_data_path = '/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/PupilData/2021-08-18_13-12-52-de913cc7/'
eye_tracking_data_path = '/home/user/Documents/Programmation/gaze_trajectory_Pupil_Xsens/PupilData/2021-08-18_13-12-52-de913cc7/'

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
Xsens_jointAngle = sio.loadmat(file_dir + file_name + 'jointAngle.mat')["jointAngle"]
Xsens_centerOfMass = sio.loadmat(file_dir + file_name + 'centerOfMass.mat')["centerOfMass"]


FLAG_SYNCHRO_PLOTS = False
FLAG_COM_PLOTS = True


############################### Load Pupil data ############################################################

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
    if float(csv_imu_read[i][0][2]) == 0:
        break
    csv_imu[i, 0] = float(csv_imu_read[i][0][2]) # timestemp
    csv_imu[i, 1] = float(csv_imu_read[i][0][3]) # gyro_x [deg/s]
    csv_imu[i, 2] = float(csv_imu_read[i][0][4])  # gyro_y [deg/s]
    csv_imu[i, 3] = float(csv_imu_read[i][0][5])  # gyro_z [deg/s]
    csv_imu[i, 4] = float(csv_imu_read[i][0][6]) * 9.81  # acceleration_x [était en G, maintenant en m/s**2]
    csv_imu[i, 5] = float(csv_imu_read[i][0][7]) * 9.81  # acceleration_y [était en G, maintenant en m/s**2]
    csv_imu[i, 6] = float(csv_imu_read[i][0][8]) * 9.81  # acceleration_z [était en G, maintenant en m/s**2]
    # csv_imu[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - time_stamps_eye_tracking)) # closest image timestemp

csv_imu = csv_imu[np.nonzero(csv_imu[:, 0])[0], :]

# 2 -> 0: imu_timestamp
# 3 -> 1: gyro_x
# 4 -> 2: gyro_y
# 5 -> 3: gyro_z
# 6 -> 4: acceleration_x
# 7 -> 5: acceleration_y
# 8 -> 6: acceleration_z


moving_average_window_size = 10 # nombre d'éléments à prendre de chaque bord
csv_imu_averaged = np.zeros((len(csv_imu), 3))
for j in range(39, 42): ############ 3
    for i in range(len(csv_imu)):
        if i < moving_average_window_size:
            csv_imu_averaged[i, j] = np.mean(csv_imu[:2 * i + 1, j + 4])
        elif i > (len(csv_imu) - moving_average_window_size - 1):
            csv_imu_averaged[i, j] = np.mean(csv_imu[-2 * (len(csv_imu) - i) + 1:, j + 4])
        else:
            csv_imu_averaged[i, j] = np.mean(csv_imu[i - moving_average_window_size : i + moving_average_window_size + 1, j + 4])


moving_average_window_size = round(moving_average_window_size / 3)
Xsens_sensorFreeAcceleration_averaged = np.zeros((len(Xsens_sensorFreeAcceleration), 3))
for j in range(3):
    for i in range(len(Xsens_sensorFreeAcceleration)):
        if i < moving_average_window_size:
            Xsens_sensorFreeAcceleration_averaged[i, j] = np.mean(Xsens_sensorFreeAcceleration[:2 * i + 1, j + 6])
        elif i > (len(Xsens_sensorFreeAcceleration) - moving_average_window_size - 1):
            Xsens_sensorFreeAcceleration_averaged[i, j] = np.mean(Xsens_sensorFreeAcceleration[-2 * (len(Xsens_sensorFreeAcceleration) - i) + 1:, j + 6])
        else:
            Xsens_sensorFreeAcceleration_averaged[i, j] = np.mean(Xsens_sensorFreeAcceleration[i - moving_average_window_size : i + moving_average_window_size + 1, j + 6])

# plt.figure()
# plt.plot(csv_imu_averaged, '-', label="averaged")
# # plt.plot(csv_imu[:, 4:7], ':', label="raw")
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(Xsens_sensorFreeAcceleration_averaged, '-', label="averaged")
# # plt.plot(Xsens_sensorFreeAcceleration[:, 6:9], ':', label="raw")
# plt.legend()
# plt.show()

csv_imu_averaged_norm = np.linalg.norm(csv_imu_averaged, axis=1)
Xsens_sensorFreeAcceleration_averaged_norm = np.linalg.norm(Xsens_sensorFreeAcceleration_averaged, axis=1)

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    plt.plot(csv_imu_averaged_norm, '-b', label="Pupil")
    plt.plot(Xsens_sensorFreeAcceleration_averaged_norm, '-r', label="Xsens")
    plt.legend()
    plt.show()

    # pied
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(Xsens_sensorFreeAcceleration_averaged[:, 39])
    axs[1].plot(Xsens_sensorFreeAcceleration_averaged[:, 40])
    axs[2].plot(Xsens_sensorFreeAcceleration_averaged[:, 41])
    plt.legend()
    plt.show()



time_vector_pupil = (csv_imu[:, 0] - csv_imu[0, 0])/1e9
time_vector_xsens = (Xsens_ms - Xsens_ms[0])/1000

time_vector_pupil_interp = np.arange(0, time_vector_pupil[-1], 1/200)
time_vector_xsens_interp = np.arange(0, time_vector_xsens[-1], 1/200)
pupil_interp = scipy.interpolate.interp1d(time_vector_pupil, csv_imu_averaged_norm)
xsens_interp = scipy.interpolate.interp1d(np.reshape(time_vector_xsens, (len(time_vector_xsens), )), Xsens_sensorFreeAcceleration_averaged_norm)
norm_accelaration_pupil = pupil_interp(time_vector_pupil_interp)
norm_accelaration_xsens = xsens_interp(time_vector_xsens_interp)

# # csv_imu_averaged_norm_sansgravite = abs(csv_imu_averaged_norm - 9.981)
# plt.figure()
# plt.plot(time_vector_pupil_interp, norm_accelaration_pupil, '-r', label='Pupil')
# # plt.plot(time_vector_imu, csv_imu_averaged_norm_sansgravite, '-r', label='Pupil')
# plt.plot(time_vector_xsens_interp, norm_accelaration_xsens, '-b', label='Xsens')
# plt.legend()
# plt.show()

norm_acceleration_xsens_croped = norm_accelaration_xsens
norm_acceleration_pupil_croped = norm_accelaration_pupil

correlation_sum = np.array([])
glide_shift = 800
if len(norm_acceleration_xsens_croped) > len(norm_acceleration_pupil_croped):  # Xsens > Pupil
    for i in range(-glide_shift , 0):

        current_correlation = np.mean(np.abs(np.correlate(norm_acceleration_xsens_croped[:len(norm_acceleration_pupil_croped)+i],
                                       norm_acceleration_pupil_croped[-i:len(norm_acceleration_xsens_croped)],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    length_diff = len(norm_acceleration_xsens_croped) - len(norm_acceleration_pupil_croped)
    for i in range(0, length_diff):

        current_correlation = np.mean(np.abs(np.correlate(norm_acceleration_xsens_croped[i : len(norm_acceleration_pupil_croped) + i],
                                       norm_acceleration_pupil_croped,
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    for i in range(length_diff, length_diff + glide_shift ):

        current_correlation = np.mean(np.abs(np.correlate(norm_acceleration_xsens_croped[i:],
                                       norm_acceleration_pupil_croped[:len(norm_acceleration_pupil_croped)-i],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

else:
    for i in range(-glide_shift , 0):
        current_correlation = np.mean(np.abs(np.correlate(norm_acceleration_pupil_croped[:len(norm_acceleration_xsens_croped)+i],
                                       norm_acceleration_xsens_croped[-i:len(norm_acceleration_pupil_croped)],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    length_diff = len(norm_acceleration_pupil_croped) - len(norm_acceleration_xsens_croped)
    for i in range(0, length_diff):
        current_correlation = np.mean(np.abs(np.correlate(norm_acceleration_pupil_croped[i : len(norm_acceleration_xsens_croped) + i],
                                       norm_acceleration_xsens_croped,
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    for i in range(length_diff, length_diff + glide_shift ):
        current_correlation = np.mean(np.abs(np.correlate(norm_acceleration_pupil_croped[i:],
                                       norm_acceleration_xsens_croped[: len(norm_acceleration_xsens_croped) -i + length_diff],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    plt.plot(correlation_sum)
    plt.show()

idx_max = np.argmax(correlation_sum) - glide_shift

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    if len(norm_acceleration_xsens_croped) > len(norm_acceleration_pupil_croped):  # Xsens > Pupil
        plt.plot(np.arange(idx_max, idx_max + len(norm_acceleration_pupil_croped)), norm_acceleration_pupil_croped, '-r', label='Pupil')
        plt.plot(norm_acceleration_xsens_croped, '-b', label='Xsens')
    else:
        plt.plot(norm_acceleration_pupil_croped, '-r', label='Pupil')
        plt.plot(np.arange(idx_max, idx_max + len(norm_acceleration_xsens_croped)), norm_acceleration_xsens_croped, '-b', label='Xsens')
    plt.legend()
    plt.show()


################################# COM ##########################################

if FLAG_COM_PLOTS:
    labels_CoM = ["X", "Y", "Z", "vitesse X", "vitesse Y", "vitesse Z", "acc X", "acc Y", "acc Z"]
    plt.figure()
    for i in range(3):
        plt.plot(Xsens_centerOfMass[:, i], label=f'{labels_CoM[i]}')
    plt.legend()
    plt.show()

    peaks_max, _ = signal.find_peaks(Xsens_centerOfMass[:, 5], prominence=(0.1, None))
    peaks_min, _ = signal.find_peaks(-Xsens_centerOfMass[:, 5], prominence=(0.1, None))

    peaks_total = np.sort(np.hstack((peaks_min, peaks_max)))

    plt.figure()
    for i in range(3, 6):
        plt.plot(time_vector_xsens, Xsens_centerOfMass[:, i], label=f'{labels_CoM[i]}')
        if i == 5:
            plt.plot(time_vector_xsens[peaks_max], Xsens_centerOfMass[peaks_max, 5], 'xg')
            plt.plot(time_vector_xsens[peaks_min], Xsens_centerOfMass[peaks_min, 5], 'xg')
            for j in range(len(peaks_total)-1):
                x_linregress = np.reshape(time_vector_xsens[peaks_total[j]:peaks_total[j+1]], (len(time_vector_xsens[peaks_total[j]:peaks_total[j+1]]), ))
                slope, intercept, _, _, _ = scipy.stats.linregress(x_linregress,
                                                       Xsens_centerOfMass[peaks_total[j] : peaks_total[j+1], 5])
                plt.plot(x_linregress,
                         intercept + slope * x_linregress,
                         '--k', alpha=0.5)
                print("slope : ", slope)

    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(time_vector_xsens, Xsens_centerOfMass[:, 5], label='acceleration CoM Z')
    plt.plot(time_vector_xsens[peaks_max], Xsens_centerOfMass[peaks_max, 5], 'xg')
    plt.plot(time_vector_xsens[peaks_min], Xsens_centerOfMass[peaks_min, 5], 'xg')
    for j in range(len(peaks_total)-1):
        x_linregress = np.reshape(time_vector_xsens[peaks_total[j]:peaks_total[j+1]], (len(time_vector_xsens[peaks_total[j]:peaks_total[j+1]]), ))
        slope, intercept, _, _, _ = scipy.stats.linregress(x_linregress,
                                               Xsens_centerOfMass[peaks_total[j] : peaks_total[j+1], 5])
        plt.plot(x_linregress,
                 intercept + slope * x_linregress,
                 '--k', alpha=0.5)
    plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, '-r', label="norm acceleration tete IMU")

    plt.legend()
    plt.show()

    plt.figure()
    for i in range(6, 9):
        plt.plot(Xsens_centerOfMass[:, i], label=f'{labels_CoM[i]}')
    plt.legend()
    plt.show()

embed()

# Animation

embed()

m = biorbd.Model(model_path)

b = bioviz.Viz(model_path)
b.load_movement(Xsens_jointAngle)
b.exec()











