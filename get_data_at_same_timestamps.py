import biorbd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.io as sio
from scipy import signal
from IPython import embed
import pandas as pd
import quaternion
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from os.path import exists
from unproject_PI_2d_pixel_gaze_estimates import pixelPoints_to_gazeAngles


def get_data_at_same_timestamps(
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
        num_joints):

    time_vector_pupil_per_move = [np.array([]) for i in range(len(start_of_move_index))]
    for i in range(len(start_of_move_index)):
        time_vector_pupil_per_move[i] = time_vector_pupil_offset[int(start_of_move_index[i]) : int(end_of_move_index[i]) + 1] # voir si c'est end+1 ou pas


    Xsens_position_per_move = [np.array([]) for i in range(len(start_of_move_index))]
    for i in range(len(start_of_move_index)):
        Xsens_position_per_move[i] = np.zeros((len(time_vector_pupil_per_move[i]), np.shape(Xsens_position)[1]))
        for j in range(np.shape(Xsens_position)[1]):
            xsens_interp_on_pupil = scipy.interpolate.interp1d(
                np.reshape(
                    time_vector_xsens[int(xsens_start_of_move_index[i]-2) : int(xsens_end_of_move_index[i])+2],
                    (len(time_vector_xsens[int(xsens_start_of_move_index[i]-2) : int(xsens_end_of_move_index[i])+2]))
                ),
                Xsens_position[int(xsens_start_of_move_index[i]-2) : int(xsens_end_of_move_index[i])+2, j])
            Xsens_position_per_move[i][:, j] = xsens_interp_on_pupil(time_vector_pupil_per_move[i])

    Xsens_orientation_per_move = [np.array([]) for _ in range(len(start_of_move_index))]
    for i in range(len(start_of_move_index)):
        Xsens_orientation_per_move[i] = np.zeros((len(time_vector_pupil_per_move[i]), 4*num_joints))
        for j in range(len(time_vector_pupil_per_move[i])):
            idx_closest = np.argmin(np.abs(time_vector_pupil_per_move[i][j] - time_vector_xsens))
            if time_vector_xsens[idx_closest] < time_vector_pupil_per_move[i][j]:
                idx_0 = idx_closest
                idx_1 = idx_closest + 1
            else:
                idx_0 = idx_closest - 1
                idx_1 = idx_closest
            t0 = time_vector_xsens[idx_0]
            t1 = time_vector_xsens[idx_1]
            for k in range(num_joints):
                quat_0 = np.quaternion(Xsens_orientation[idx_0, 4*k],
                                       Xsens_orientation[idx_0, 4*k+1],
                                       Xsens_orientation[idx_0, 4*k+2],
                                       Xsens_orientation[idx_0, 4*k+3])
                quat_1 = np.quaternion(Xsens_orientation[idx_1, 4*k],
                                       Xsens_orientation[idx_1, 4*k+1],
                                       Xsens_orientation[idx_1, 4*k+2],
                                       Xsens_orientation[idx_1, 4*k+3])
                relative_time = (time_vector_pupil_per_move[i][j] - t0)/(t1 - t0)
                interp_quat =  quaternion.slerp_evaluate(quat_0, quat_1, relative_time).components

                Xsens_orientation_per_move[i][j, 4 * k:4 * (k + 1)] = interp_quat

    Xsens_CoM_per_move = [np.array([]) for i in range(len(start_of_move_index))]
    for i in range(len(start_of_move_index)):
        Xsens_CoM_per_move[i] = np.zeros((len(time_vector_pupil_per_move[i]), 9))
        for j in range(9):
            xsens_interp_on_pupil = scipy.interpolate.interp1d(
                np.reshape(
                time_vector_xsens[int(xsens_start_of_move_index[i])-2 : int(xsens_end_of_move_index[i])+2],
                len(time_vector_xsens[int(xsens_start_of_move_index[i])-2 : int(xsens_end_of_move_index[i])+2])),
                Xsens_centerOfMass[int(xsens_start_of_move_index[i])-2 : int(xsens_end_of_move_index[i])+2, j])
            Xsens_CoM_per_move[i][:, j] = xsens_interp_on_pupil(time_vector_pupil_per_move[i])

    elevation_pupil_pixel = csv_eye_tracking[:, 1]
    azimuth_pupil_pixel = csv_eye_tracking[:, 2]
    elevation, azimuth = pixelPoints_to_gazeAngles(elevation_pupil_pixel, azimuth_pupil_pixel, SCENE_CAMERA_SERIAL_NUMBER)

    elevation_per_move = [np.array([]) for i in range(len(start_of_move_index))]
    azimuth_per_move = [np.array([]) for i in range(len(start_of_move_index))]
    for i in range(len(start_of_move_index)):
        elevation_per_move[i] = elevation[int(start_of_move_index[i]) : int(end_of_move_index[i]) + 1] # voir si c'est end+1 ou pas
        azimuth_per_move[i] = azimuth[int(start_of_move_index[i]): int(end_of_move_index[i]) + 1]  # voir si c'est end+1 ou pas


    plt.figure()
    plt.plot(time_vector_pupil_offset, elevation, '-r', label='elevation')
    plt.plot(time_vector_pupil_offset, azimuth, '-b', label='azimuth')
    for i in range(len(time_vector_pupil_per_move)):
        plt.plot(time_vector_pupil_per_move[i], elevation_per_move [i], '-m')
        plt.plot(time_vector_pupil_per_move[i], azimuth_per_move[i], '-c')
    plt.show()

    return time_vector_pupil_per_move, Xsens_orientation_per_move, Xsens_position_per_move, Xsens_CoM_per_move, elevation_per_move, azimuth_per_move

























