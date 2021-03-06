import biorbd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.io as sio
from scipy import signal
from IPython import embed
import pandas as pd
# import biorbd
import bioviz
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from os.path import exists
from unproject_PI_2d_pixel_gaze_estimates import pixelPoints_to_gazeAngles

############################### Load general data ##############################################
file_name = 'Test_17032021-007/'

if exists("/home/user"):
    home_path = "/home/user"
    file_dir = '/home/user/Documents/Eye-tracking/XsensData/'
elif exists("/home/fbailly"):
    home_path = "/home/fbailly"
    file_dir = "/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData/"

csv_name = home_path + "/Documents/Programmation/rectangle-labelling/Trials_name_mapping.csv"
csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')

movie_path = home_path + "/Documents/Eye-tracking/PupilData/points_labeled/"

# for i_trial in range(len(csv_table)):
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
                'jRightT4Shoulder???', 'jRightT4Shoulder???', 'jRightT4Shoulder???', 'jRightShoulder_x', 'jRightShoulder_y',
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

csv_gaze_read = np.char.split(pd.read_csv(filename_gaze, sep='\t').values.astype('str'), sep=',')
csv_imu_read = np.char.split(pd.read_csv(filename_imu, sep='\t').values.astype('str'), sep=',')
timestamp_image = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')

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

# plt.figure()
# plt.plot(np.abs(csv_eye_tracking[:, 0] - time_stamps_eye_tracking[600]))
# plt.title("csv_eye_tracking[:, 0] - time_stamps_eye_tracking[600]")
# plt.show()

zeros_clusters_index = curent_AOI_label["Not an acrobatics"][:-1] - curent_AOI_label["Not an acrobatics"][1:]
zeros_clusters_index = np.hstack((0, zeros_clusters_index))
end_of_move_index_image = np.where(zeros_clusters_index == -1)[0].tolist()
start_of_move_index_image = np.where(zeros_clusters_index == 1)[0].tolist()

end_of_move_index = time_stamps_eye_tracking_index_on_pupil[end_of_move_index_image]
start_of_move_index = time_stamps_eye_tracking_index_on_pupil[start_of_move_index_image]

# plt.figure()
# plt.plot(time_stamps_eye_tracking_index_on_pupil)
# plt.plot(end_of_move_index_image, time_stamps_eye_tracking_index_on_pupil[end_of_move_index_image], 'xr')
# plt.plot(start_of_move_index_image, time_stamps_eye_tracking_index_on_pupil[start_of_move_index_image], 'xg')
# plt.title("timestaps + start(xr) and end(xg) of move")
# plt.show()

# movie_path = home_path + "/Documents/Eye-tracking/PupilData/undistorted_videos/"
# movie_file = movie_path + movie_name + "_undistorted_images.pkl"
# file = open(movie_file, "rb")
# frames = pickle.load(file)
# num_frames = len(frames)
#
# plt.figure()
# plt.imshow(frames[int(start_of_move_index_image[0])])
# plt.title("Take-off frame")
# plt.show()
#
# plt.figure()
# plt.imshow(frames[int(end_of_move_index_image[0])])
# plt.title("Landing frame")
# plt.show()

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
    csv_imu[i, 4] = float(csv_imu_read[i][0][6]) * 9.81  # acceleration_x [??tait en G, maintenant en m/s**2]
    csv_imu[i, 5] = float(csv_imu_read[i][0][7]) * 9.81  # acceleration_y [??tait en G, maintenant en m/s**2]
    csv_imu[i, 6] = float(csv_imu_read[i][0][8]) * 9.81  # acceleration_z [??tait en G, maintenant en m/s**2]
    # csv_imu[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - time_stamps_eye_tracking)) # closest image timestemp

csv_imu = csv_imu[np.nonzero(csv_imu[:, 0])[0], :]

# 2 -> 0: imu_timestamp
# 3 -> 1: gyro_x
# 4 -> 2: gyro_y
# 5 -> 3: gyro_z
# 6 -> 4: acceleration_x
# 7 -> 5: acceleration_y
# 8 -> 6: acceleration_z

################################################# Animate JCS ##########################################################

def animate(Xsens_position, CoM_trajectory, elevation, azimuth, links, output_file_name, max_frame=0):

    # Make sure eyes are at the right place
    # Plot walls

    def Xsens_quat_to_orientation(Xsens_orientation_i, Xsens_position_i, elevation, azimuth, i_line):
        Xsens_position_calculated = [np.zeros((6, )) for _ in range(num_joints)]
        Quat_normalized = Xsens_orientation_i[4*i_line:4*(i_line+1)] / np.linalg.norm(Xsens_orientation_i[4*i_line:4*(i_line+1)])
        Quat = biorbd.Quaternion(Quat_normalized[0], Quat_normalized[1], Quat_normalized[2], Quat_normalized[3])
        RotMat = biorbd.Quaternion.toMatrix(Quat).to_array()
        Xsens_position_calculated[i_line][:3] = Xsens_position_i[3*i_line:3*(i_line+1)]
        Xsens_position_calculated[i_line][3:] = RotMat @ np.array([0, 0, 0.1]) + Xsens_position_i[3*i_line:3*(i_line+1)]
        if i_line == 6:
            eye_position = RotMat @ np.array([eye_position_depth, 0, eye_position_height]) + Xsens_position_i[3 * i_line:3 * (i_line + 1)]
            gaze_rotMat = biorbd.Rotation_fromEulerAngles(np.array([azimuth, elevation]), 'xz')
            gaze_orientation = gaze_rotMat @ RotMat @ np.array([3, 0, 0]) + eye_position
        else:
            eye_position = np.array([0, 0, 0])
            gaze_orientation = np.array([0, 0, 0])
        return Xsens_position_calculated, eye_position, gaze_orientation


    def update(i, Xsens_position, CoM_trajectory, lines, CoM_point, line_orientation, eyes_point, elevation, azimuth, links):

        CoM_point[0][0].set_data(np.array([CoM_trajectory[i, 0]]), np.array([CoM_trajectory[i, 1]]))
        CoM_point[0][0].set_3d_properties(np.array([CoM_trajectory[i, 2]]))

        for i_line, line in enumerate(lines):
            line[0].set_data(np.array([Xsens_position[i][3 * links[i_line, 0]], Xsens_position[i][3 * links[i_line, 1]] ]), np.array([Xsens_position[i][3 * links[i_line, 0] + 1], Xsens_position[i][3 * links[i_line, 1] + 1] ]))
            line[0].set_3d_properties(np.array([Xsens_position[i][3 * links[i_line, 0] + 2], Xsens_position[i][3 * links[i_line, 1] + 2] ]))

        for i_line, line in enumerate(line_orientation[:7]): # enumerate(line_orientation):
            Xsens_position_calculated, eye_position, gaze_orientation = Xsens_quat_to_orientation(Xsens_orientation[i, :], Xsens_position[i, :], elevation[i], azimuth[i], i_line)
            line[0].set_data(np.array([Xsens_position_calculated[i_line][0], Xsens_position_calculated[i_line][3]]), np.array([Xsens_position_calculated[i_line][1], Xsens_position_calculated[i_line][4]]))
            line[0].set_3d_properties(np.array([Xsens_position_calculated[i_line][2], Xsens_position_calculated[i_line][5]]))
            if i_line == 6:
                eyes_point[0][0].set_data(np.array([eye_position[0]]), np.array([eye_position[1]]))
                eyes_point[0][0].set_3d_properties(np.array([eye_position[2]]))

        return lines

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_box_aspect([1, 1, 1])

    CoM_point = [ax.plot(0, 0, 0, '.r')]
    eyes_point = [ax.plot(0, 0, 0, '.c')]
    lines = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-k') for _ in range(len(links))]
    line_orientation = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-m') for _ in range(num_joints)]

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    if max_frame == 0:
        frame_range = range(len(Xsens_position))
    else:
        frame_range = range(max_frame)

    anim = animation.FuncAnimation(fig, update, frames=frame_range, fargs=(Xsens_position, CoM_trajectory, lines, CoM_point, line_orientation, eyes_point, links), blit=False)

    anim.save(output_file_name, fps=60, extra_args=['-vcodec', 'libx264'])
    plt.show()

    return

# output_file_name = home_path + f'/Documents/Programmation/rectangle-labelling/output/Results/{Xsens_Subject_name[0]}/{movie_name}_animation_no_level.mp4'
# animate(Xsens_position, Xsens_centerOfMass, elevation, azimuth, links, output_file_name, 50)

###############################

FLAG_SYNCHRO_PLOTS =  True #False #
FLAG_COM_PLOTS = False # True #


moving_average_window_size = 10 # nombre d'??l??ments ?? prendre de chaque bord
csv_imu_averaged = np.zeros((len(csv_imu), 3))
for j_idx, j in enumerate(range(3)):
    for i in range(len(csv_imu)):
        if i < moving_average_window_size:
            csv_imu_averaged[i, j_idx] = np.mean(csv_imu[:2 * i + 1, j + 4])
        elif i > (len(csv_imu) - moving_average_window_size - 1):
            csv_imu_averaged[i, j_idx] = np.mean(csv_imu[-2 * (len(csv_imu) - i) + 1:, j + 4])
        else:
            csv_imu_averaged[i, j_idx] = np.mean(csv_imu[i - moving_average_window_size : i + moving_average_window_size + 1, j + 4])


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

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    plt.plot(csv_imu_averaged, '-', label="averaged")
    plt.plot(csv_imu[:, 4:7], ':', label="raw")
    for i in range(len(start_of_move_index)):
        if i == 0:
            plt.plot(np.ones((2, )) * start_of_move_index[i], np.array([np.min(csv_imu_averaged), np.max(csv_imu_averaged)]), '-k', label='Start of movement')
            plt.plot(np.ones((2,)) * end_of_move_index[i], np.array([np.min(csv_imu_averaged), np.max(csv_imu_averaged)]), '-k', label='End of movement')
        else:
            plt.plot(np.ones((2,)) * start_of_move_index[i], np.array([np.min(csv_imu_averaged), np.max(csv_imu_averaged)]), '-k')
            plt.plot(np.ones((2,)) * end_of_move_index[i], np.array([np.min(csv_imu_averaged), np.max(csv_imu_averaged)]), '-k')
    plt.legend()
    plt.title('Pupil')
    plt.show()

    plt.figure()
    plt.plot(Xsens_sensorFreeAcceleration_averaged, '-', label="averaged")
    plt.plot(Xsens_sensorFreeAcceleration[:, 6:9], ':', label="raw")
    plt.legend()
    plt.title('Xsens')
    plt.show()

csv_imu_averaged_norm = np.linalg.norm(csv_imu_averaged, axis=1) + 9.81
Xsens_sensorFreeAcceleration_averaged_norm = np.linalg.norm(Xsens_sensorFreeAcceleration_averaged, axis=1)

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    plt.plot(csv_imu_averaged_norm, '-b', label="Pupil")
    plt.plot(Xsens_sensorFreeAcceleration_averaged_norm, '-r', label="Xsens")
    plt.legend()
    plt.title('Raw')
    plt.show()

    # # pied
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(Xsens_sensorFreeAcceleration_averaged[:, 39])
    # axs[1].plot(Xsens_sensorFreeAcceleration_averaged[:, 40])
    # axs[2].plot(Xsens_sensorFreeAcceleration_averaged[:, 41])
    # plt.legend()
    # plt.show()



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

correlation_sum = np.array([])
glide_shift = 800
if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
    for i in range(-glide_shift , 0):

        current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_xsens[:len(norm_accelaration_pupil)+i],
                                       norm_accelaration_pupil[-i:len(norm_accelaration_xsens)],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    length_diff = len(norm_accelaration_xsens) - len(norm_accelaration_pupil)
    for i in range(0, length_diff):

        current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_xsens[i : len(norm_accelaration_pupil) + i],
                                       norm_accelaration_pupil,
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    for i in range(length_diff, length_diff + glide_shift ):

        current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_xsens[i:],
                                       norm_accelaration_pupil[:len(norm_accelaration_pupil)-i],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

else:
    for i in range(-glide_shift , 0):
        current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_pupil[:len(norm_accelaration_xsens)+i],
                                       norm_accelaration_xsens[-i:len(norm_accelaration_pupil)],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    length_diff = len(norm_accelaration_pupil) - len(norm_accelaration_xsens)
    for i in range(0, length_diff):
        current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_pupil[i : len(norm_accelaration_xsens) + i],
                                       norm_accelaration_xsens,
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

    for i in range(length_diff, length_diff + glide_shift ):
        current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_pupil[i:],
                                       norm_accelaration_xsens[: len(norm_accelaration_xsens) -i + length_diff],
                                       mode='valid')))
        correlation_sum = np.hstack((correlation_sum, current_correlation))

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    plt.plot(correlation_sum)
    plt.title('Correlation')
    plt.show()

idx_max = np.argmax(correlation_sum) - glide_shift
glide_shift_time = idx_max * 1/200


if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
        plt.plot(np.arange(idx_max, idx_max + len(norm_accelaration_pupil)), norm_accelaration_pupil, '-r', label='Pupil')
        plt.plot(norm_accelaration_xsens, '-b', label='Xsens')
    else:
        plt.plot(norm_accelaration_pupil, '-r', label='Pupil')
        plt.plot(np.arange(idx_max, idx_max + len(norm_accelaration_xsens)), norm_accelaration_xsens, '-b', label='Xsens')
    plt.legend()
    plt.title('Synchro interp')
    plt.show()

    plt.figure()
    if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
        plt.plot(time_vector_pupil + glide_shift_time, csv_imu_averaged_norm, '-r', label='Pupil')
        plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, '-b', label='Xsens')
    else:
        plt.plot(time_vector_pupil, csv_imu_averaged_norm, '-r', label='Pupil')
        plt.plot(time_vector_xsens + glide_shift_time, Xsens_sensorFreeAcceleration_averaged_norm, '-b', label='Xsens')
    plt.legend()
    plt.title('Synchro pas interp')
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



if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
    xsens_time_slided = np.reshape(time_vector_xsens - glide_shift_time, len(time_vector_xsens))
else:
    xsens_time_slided = np.reshape(time_vector_xsens + glide_shift_time, len(time_vector_xsens))


if xsens_time_slided[0] > time_vector_pupil[0]:
    arg_min_time = np.argmin(np.abs(xsens_time_slided[0] - time_vector_pupil)) + 5
    start_of_move_index_updated = start_of_move_index - arg_min_time
else:
    arg_min_time = 0 + 5
if xsens_time_slided[-1] < time_vector_pupil[-1]:
    arg_max_time = np.argmin(np.abs(xsens_time_slided[-1] - time_vector_pupil)) - 5
else:
    arg_max_time = len(time_vector_pupil) - 5

start_of_move_index_updated = np.asarray(start_of_move_index - arg_min_time, np.int64)
end_of_move_index_updated = np.asarray(end_of_move_index - arg_min_time, np.int64)

time_vector_pupil_chopped = time_vector_pupil[arg_min_time:arg_max_time]
# chop pupil data avec ces args min max la aussi pour avoir exactement les meme time points avec Xsens ###########
elevation_pupil_pixel_choped = csv_eye_tracking[arg_min_time:arg_max_time, 1]
azimuth_pupil_pixel_choped = csv_eye_tracking[arg_min_time:arg_max_time, 2]

Xsens_position_on_pupil = np.zeros((len(time_vector_pupil_chopped), np.shape(Xsens_position)[1]))
for i in range(np.shape(Xsens_position)[1]):
    xsens_interp_on_pupil = scipy.interpolate.interp1d(xsens_time_slided, Xsens_position[:, i])
    Xsens_position_on_pupil[:, i] = xsens_interp_on_pupil(time_vector_pupil_chopped)

Xsens_CoM_on_pupil = np.zeros((len(time_vector_pupil_chopped), 9))
for i in range(9):
    xsens_interp_on_pupil = scipy.interpolate.interp1d(xsens_time_slided, Xsens_centerOfMass[:, i])
    Xsens_CoM_on_pupil[:, i] = xsens_interp_on_pupil(time_vector_pupil_chopped)

xsens_interp_on_pupil = scipy.interpolate.interp1d(xsens_time_slided, Xsens_sensorFreeAcceleration_averaged_norm)
Xsens_sensorFreeAcceleration_averaged_norm_on_pupil = xsens_interp_on_pupil(time_vector_pupil_chopped)

if FLAG_SYNCHRO_PLOTS:
    plt.figure()
    plt.plot(time_vector_pupil_chopped, csv_imu_averaged_norm[arg_min_time : arg_max_time], label='Pupil choped')
    plt.plot(time_vector_pupil_chopped, Xsens_sensorFreeAcceleration_averaged_norm_on_pupil, label='Xsens interpolated on pupil')
    for i in range(len(start_of_move_index_updated)):
        if i == 0:
            plt.plot(np.ones((2, )) * time_vector_pupil_chopped[start_of_move_index_updated[i]], np.array([0, np.max(csv_imu_averaged_norm[arg_min_time : arg_max_time])]), '--g', label='Start of movement')
            plt.plot(np.ones((2,)) * time_vector_pupil_chopped[end_of_move_index_updated[i]], np.array([0, np.max(csv_imu_averaged_norm[arg_min_time: arg_max_time])]), '--r', label='End of movement')
        else:
            plt.plot(np.ones((2, )) * time_vector_pupil_chopped[start_of_move_index_updated[i]], np.array([0, np.max(csv_imu_averaged_norm[arg_min_time : arg_max_time])]), '--g')
            plt.plot(np.ones((2,)) * time_vector_pupil_chopped[end_of_move_index_updated[i]], np.array([0, np.max(csv_imu_averaged_norm[arg_min_time: arg_max_time])]), '--r')
    plt.legend()
    plt.show()

Xsens_position_on_pupil_no_level = np.zeros(np.shape(Xsens_position_on_pupil))
Xsens_CoM_on_pupil_no_level = np.zeros((np.shape(Xsens_CoM_on_pupil)[0], 3))
for i in range(np.shape(Xsens_position_on_pupil)[0]):
    Pelvis_position = Xsens_position_on_pupil[i, :3]
    for j in range(num_joints):
        Xsens_position_on_pupil_no_level[i, 3*j:3*(j+1)] = Xsens_position_on_pupil[i, 3*j:3*(j+1)] - Pelvis_position + np.array([0, 0, hip_height])
    Xsens_CoM_on_pupil_no_level[i, :] = Xsens_CoM_on_pupil[i, :3]  - Pelvis_position + np.array([0, 0, hip_height])

if FLAG_COM_PLOTS:
    plt.figure()
    plt.plot(Xsens_position_on_pupil[:, 0], label='pelvis X')
    plt.plot(Xsens_position_on_pupil[:, 1], label='pelvis Y')
    plt.plot(Xsens_position_on_pupil[:, 2], label='pelvis Z')
    plt.plot(Xsens_CoM_on_pupil[:, 0], label='CoM X')
    plt.plot(Xsens_CoM_on_pupil[:, 1], label='CoM Y')
    plt.plot(Xsens_CoM_on_pupil[:, 2], label='CoM Z')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(Xsens_position_on_pupil_no_level[:, 0], label='pelvis X _no_level')
    plt.plot(Xsens_position_on_pupil_no_level[:, 1], label='pelvis Y _no_level')
    plt.plot(Xsens_position_on_pupil_no_level[:, 2], label='pelvis Z _no_level')
    plt.plot(Xsens_CoM_on_pupil_no_level[:, 0], label='CoM X _no_level')
    plt.plot(Xsens_CoM_on_pupil_no_level[:, 1], label='CoM Y _no_level')
    plt.plot(Xsens_CoM_on_pupil_no_level[:, 2], label='CoM Z _no_level')
    plt.legend()
    plt.show()


##################################### Gaze  angle ################################################

elevation, azimuth = pixelPoints_to_gazeAngles(elevation_pupil_pixel_choped, azimuth_pupil_pixel_choped)

output_file_name = home_path + f'/Documents/Programmation/rectangle-labelling/output/Results/{Xsens_Subject_name[0]}/{movie_name}_animation_no_level.mp4'
animate(Xsens_position_on_pupil_no_level, Xsens_CoM_on_pupil_no_level, elevation, azimuth, links, output_file_name, 0)
for i_move in range(len(move_names)):
    start_index = start_of_move_index_updated[i_move]
    end_index = end_of_move_index_updated[i_move]
    start_time = time_vector_pupil_chopped[start_index]
    end_time = time_vector_pupil_chopped[end_index]

    ToF_imove = end_time - start_time
    Xsens_CoM_on_pupil_no_level_imove = Xsens_CoM_on_pupil_no_level[start_index : end_index]
    airborn_time = time_vector_pupil_chopped[start_index:end_index] - time_vector_pupil_chopped[start_index]

    CoM_initial_position = np.array([0, 0, hip_height]) - np.array([0, 0, Xsens_CoM_on_pupil_no_level[start_index, 2]])
    CoM_final_position = np.array([0, 0, hip_height]) - np.array([0, 0, Xsens_CoM_on_pupil_no_level[end_index, 2]])
    CoM_initial_velocity = (CoM_final_position - CoM_initial_position - 0.5*-9.81*ToF_imove**2) / ToF_imove

    CoM_trajectory_imove = np.zeros((end_index - start_index, 3))
    CoM_trajectory_imove[:, 2] = CoM_initial_position[2] + CoM_initial_velocity[2]*airborn_time + 0.5*-9.81*airborn_time**2

    hip_trajectory_imove = CoM_trajectory_imove - Xsens_CoM_on_pupil_no_level_imove

    Xsens_position_on_pupil_no_level_CoM_corrected = np.zeros(np.shape(Xsens_position_on_pupil_no_level[start_index : end_index]))
    for i in range(np.shape(Xsens_position_on_pupil_no_level_CoM_corrected)[0]):
        for j in range(num_joints):
            Xsens_position_on_pupil_no_level_CoM_corrected[i, 3*j:3*(j+1)] = Xsens_position_on_pupil_no_level[start_index+i,3*j:3*(j+1)] + hip_trajectory_imove[i, :]

    output_file_name = home_path + f'/Documents/Programmation/rectangle-labelling/output/Results/{Xsens_Subject_name[0]}/{move_names[i_move]}/{movie_name}_animation_{i_move}.mp4'
    animate(Xsens_position_on_pupil_no_level_CoM_corrected, CoM_trajectory_imove, elevation, azimuth, links, output_file_name, 0)
    embed()






















