import biorbd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.io as sio
from scipy import signal
from IPython import embed
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from os.path import exists
from unproject_PI_2d_pixel_gaze_estimates import pixelPoints_to_gazeAngles
from sync_jump import sync_jump
from CoM_transfo import CoM_transfo
from gaze_orientation import gaze_orientation
from get_data_at_same_timestamps import get_data_at_same_timestamps
from animate_JCS import animate

############################### Load general data ##############################################
file_name = 'Test_17032021-007/'

if exists("/home/user"):
    home_path = "/home/user"
    file_dir = '/home/user/Documents/Eye-tracking/XsensData/'
elif exists("/home/fbailly"):
    home_path = "/home/fbailly"
    # file_dir = "/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData/"
    file_dir = '/home/fbailly/Documents/Eye-tracking/XsensData/'

csv_name = home_path + "/Documents/Programmation/rectangle-labelling/Trials_name_mapping.csv"
csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')

movie_path = home_path + "/Documents/Eye-tracking/PupilData/points_labeled/"

# for i_trial in range(len(csv_table)):
######################################
i_trial = 0
movie_name = csv_table[i_trial][0][6].replace('.', '_')
gaze_position_labels = movie_path + movie_name + "_labeling_points.pkl"
out_path = home_path + '/Documents/Programmation/rectangle-labelling/output/Results'
subject_name = csv_table[i_trial][0][0]
move_names = csv_table[i_trial][0][1].split(" ")
move_orientation = [int(x) for x in csv_table[i_trial][0][2].split(" ")]
eye_tracking_data_path = home_path + '/Documents/Eye-tracking/PupilData/CloudExport/' + csv_table[i_trial][0][5] + '/'
subject_expertise = csv_table[i_trial][0][8]


global eye_position_height, eye_position_depth
csv_name = home_path + f"/Documents/Programmation/rectangle-labelling/output/Results/{subject_name}/{subject_name}_anthropo.csv"
csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')
hip_height = float(csv_table[2][0][1])
eye_position_height = float(csv_table[11][0][1])
eye_position_depth = float(csv_table[12][0][1])


############################### Load Xsens data ##############################################

segment_Quat_label = [ 'Pelvis_w', 'Pelvis_i', 'Pelvis_j', 'Pelvis_k', 'L5_w', 'L5_i', 'L5_j', 'L5_k', 'L3_w', 'L3_i',
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

links = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [4, 7],
                  [7, 8],
                  [8, 9],
                  [9, 10],
                  [4, 11],
                  [11, 12],
                  [12, 13],
                  [13, 14],
                  [0, 15],
                  [15, 16],
                  [16, 17],
                  [17, 18],
                  [0, 19],
                  [19, 20],
                  [20, 21],
                  [21, 22]])


Xsens_Subject_name = sio.loadmat(file_dir + file_name + 'Subject_name.mat')["Subject_name"]
Xsens_Move_name = sio.loadmat(file_dir + file_name + 'Move_name.mat')["Move_name"]
Xsens_Subject_name = sio.loadmat(file_dir + file_name + 'Subject_name.mat')["Subject_name"]
Xsens_frameRate = sio.loadmat(file_dir + file_name + 'frameRate.mat')["frameRate"]
Xsens_time = sio.loadmat(file_dir + file_name + 'time.mat')["time"]
Xsens_index = sio.loadmat(file_dir + file_name + 'index.mat')["index"]
Xsens_ms = sio.loadmat(file_dir + file_name + 'ms.mat')["ms"]

Xsens_position = sio.loadmat(file_dir + file_name + 'position.mat')["position"]
Xsens_orientation = sio.loadmat(file_dir + file_name + 'orientation.mat')["orientation"]
Xsens_velocity = sio.loadmat(file_dir + file_name + 'velocity.mat')["velocity"]
# Xsens_acceleration = sio.loadmat(file_dir + file_name + 'acceleration.mat')["acceleration"]
# Xsens_angularVelocity = sio.loadmat(file_dir + file_name + 'angularVelocity.mat')["angularVelocity"]
# Xsens_angularAcceleration = sio.loadmat(file_dir + file_n_frontame + 'angularAcceleration.mat')["angularAcceleration"]
Xsens_sensorFreeAcceleration = sio.loadmat(file_dir + file_name + 'sensorFreeAcceleration.mat')["sensorFreeAcceleration"]
# Xsens_sensorOrientation = sio.loadmat(file_dir + file_name + 'sensorOrientation.mat')["sensorOrientation"]
Xsens_jointAngle = sio.loadmat(file_dir + file_name + 'jointAngle.mat')["jointAngle"]
Xsens_centerOfMass = sio.loadmat(file_dir + file_name + 'centerOfMass.mat')["centerOfMass"]
Xsens_global_JCS_positions = sio.loadmat(file_dir + file_name + 'global_JCS_positions.mat')["global_JCS_positions"]

num_joints = int(round(np.shape(Xsens_position)[1])/3)


############################### Load Pupil data ##############################################

file = open(gaze_position_labels, "rb")
points_labels, active_points, curent_AOI_label, csv_eye_tracking = pickle.load(file)

filename_gaze = eye_tracking_data_path + 'gaze.csv'
filename_imu = eye_tracking_data_path + 'imu.csv'
filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'
filename_info = eye_tracking_data_path + 'info.json'

csv_gaze_read = np.char.split(pd.read_csv(filename_gaze, sep='\t').values.astype('str'), sep=',')
csv_imu_read = np.char.split(pd.read_csv(filename_imu, sep='\t').values.astype('str'), sep=',')
timestamp_image = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')
info = np.char.split(pd.read_csv(filename_info, sep='\t').values.astype('str'), sep=',')
serial_number_str = info[15][0][0]
num_quote = 0
for pos, char in enumerate(serial_number_str):
    if char == '"':
        num_quote += 1
        if num_quote == 3:
            SCENE_CAMERA_SERIAL_NUMBER = serial_number_str[pos+1:pos+6]
            break

csv_eye_tracking = np.zeros((len(csv_gaze_read), 7))
for i in range(len(csv_gaze_read)):
    csv_eye_tracking[i, 0] = float(csv_gaze_read[i][0][2])  # timestemp
    csv_eye_tracking[i, 1] = int(round(float(csv_gaze_read[i][0][3])))  # pos_x
    csv_eye_tracking[i, 2] = int(round(float(csv_gaze_read[i][0][4])))  # pos_y
    csv_eye_tracking[i, 3] = float(csv_gaze_read[i][0][5])  # confidence

time_stamps_eye_tracking = np.zeros((len(timestamp_image),))
time_stamps_eye_tracking_index_on_pupil = np.zeros((len(timestamp_image),))
for i in range(len(timestamp_image)):
    time_stamps_eye_tracking_index_on_pupil[i] = np.argmin(np.abs(csv_eye_tracking[:, 0] - float(timestamp_image[i][0][2])))

zeros_clusters_index = curent_AOI_label["Not an acrobatics"][:-1] - curent_AOI_label["Not an acrobatics"][1:]
zeros_clusters_index = np.hstack((0, zeros_clusters_index))

end_of_cluster_index_image = np.where(zeros_clusters_index == -1)[0].tolist()
start_of_cluster_index_image = np.where(zeros_clusters_index == 1)[0].tolist()

start_of_move_index_image = []
end_of_move_index_image = []
start_of_jump_index_image = []
end_of_jump_index_image = []
for i in range(len(start_of_cluster_index_image)):
    if curent_AOI_label["Jump"][start_of_cluster_index_image[i] + 1] == 1:
        start_of_jump_index_image += [start_of_cluster_index_image[i]]
        end_of_jump_index_image += [end_of_cluster_index_image[i]]
    else:
        start_of_move_index_image += [start_of_cluster_index_image[i]]
        end_of_move_index_image += [end_of_cluster_index_image[i]]


end_of_move_index = time_stamps_eye_tracking_index_on_pupil[end_of_move_index_image]
start_of_move_index = time_stamps_eye_tracking_index_on_pupil[start_of_move_index_image]
end_of_jump_index = time_stamps_eye_tracking_index_on_pupil[end_of_jump_index_image]
start_of_jump_index = time_stamps_eye_tracking_index_on_pupil[start_of_jump_index_image]


# 2 -> 0: gaze_timestamp
# 3 -> 1: norm_pos_x
# 4 -> 2: norm_pos_y
# 5 -> 3: confidence
# 4: closest image time_stamp
# 5: pos_x_bedFrame -> computed from labeling and distortion
# 6: pos_y_bedFrame -> computed from labeling and distortion

csv_imu = np.zeros((len(csv_imu_read), 7))
for i in range(len(csv_imu_read)):
    if float(csv_imu_read[i][0][2]) == 0:
        break
    csv_imu[i, 0] = float(csv_imu_read[i][0][2])  # timestemp
    csv_imu[i, 1] = float(csv_imu_read[i][0][3])  # gyro_x [deg/s]
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



######################################################################################################################


output_file_name = home_path + f'/Documents/Programmation/rectangle-labelling/output/Results/{Xsens_Subject_name[0]}/{movie_name}_animation_no_level.mp4'
animate(Xsens_orientation, [Xsens_position], [Xsens_centerOfMass], [np.zeros((len(Xsens_position)))], [np.zeros((len(Xsens_position)))], eye_position_height, eye_position_depth, links, num_joints, output_file_name, 250)

FLAG_SYNCHRO_PLOTS =  True #False #
FLAG_COM_PLOTS = False # True #

xsens_start_of_jump_index, xsens_end_of_jump_index, xsens_start_of_move_index, xsens_end_of_move_index, time_vector_xsens, time_vector_pupil_offset = sync_jump(Xsens_sensorFreeAcceleration, start_of_jump_index, end_of_jump_index, start_of_move_index, end_of_move_index, FLAG_SYNCHRO_PLOTS, csv_imu, Xsens_ms)

time_vector_pupil_per_move, Xsens_orientation_per_move, Xsens_position_per_move, Xsens_CoM_per_move, elevation_per_move, azimuth_per_move = get_data_at_same_timestamps(
    Xsens_orientation,
    Xsens_position,
    xsens_start_of_move_index,
    xsens_end_of_move_index,
    time_vector_xsens,
    start_of_move_index,
    end_of_move_index,
    time_vector_pupil_offset,
    csv_eye_tracking,
    Xsens_centerOfMass,
    SCENE_CAMERA_SERIAL_NUMBER,
    num_joints)

Xsens_position_no_level_CoM_corrected_per_move, CoM_trajectory_per_move = CoM_transfo(
    time_vector_pupil_per_move,
    Xsens_position_per_move,
    Xsens_CoM_per_move,
    num_joints,
    hip_height,
    FLAG_COM_PLOTS)

output_file_name = home_path + f'/Documents/Programmation/rectangle-labelling/output/Results/{Xsens_Subject_name[0]}/{movie_name}_animation_no_level.mp4'
animate(Xsens_orientation_per_move, Xsens_position_no_level_CoM_corrected_per_move, CoM_trajectory_per_move, elevation_per_move, azimuth_per_move, eye_position_height, eye_position_depth, links, num_joints, output_file_name, 0)






















