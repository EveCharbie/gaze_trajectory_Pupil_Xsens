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


def CoM_transfo(time_vector_pupil_per_move, Xsens_position_per_move, Xsens_CoM_per_move, num_joints, hip_height, FLAG_COM_PLOTS):

    if FLAG_COM_PLOTS:
        labels_CoM = ["X", "Y", "Z", "vitesse X", "vitesse Y", "vitesse Z", "acc X", "acc Y", "acc Z"]
        plt.figure()
        for j in range(len(time_vector_pupil_per_move)):
            for i in range(3):
                plt.plot(Xsens_CoM_per_move[j][:, i], label=f'{labels_CoM[i]} {j}th move')
        plt.legend()
        plt.show()

    Xsens_position_no_level = [np.zeros(np.shape(Xsens_position_per_move[i])) for i in range(len(time_vector_pupil_per_move))]
    Xsens_CoM_no_level = [np.zeros((len(time_vector_pupil_per_move[i]), 3)) for i in range(len(time_vector_pupil_per_move))]
    for j in range(len(time_vector_pupil_per_move)):
        for i in range(np.shape(Xsens_position_per_move[j])[0]):
            Pelvis_position = Xsens_position_per_move[j][i, :3]
            for k in range(num_joints):
                Xsens_position_no_level[j][i, 3 * k:3 * (k + 1)] = Xsens_position_per_move[j][i, 3 * k:3 * (k + 1)] - Pelvis_position + np.array([0, 0, hip_height])
            Xsens_CoM_no_level[j][i, :] = Xsens_CoM_per_move[j][i, :3] - Pelvis_position + np.array([0, 0, hip_height])

    CoM_trajectory = [np.zeros(np.shape(Xsens_CoM_per_move[i])) for i in range(len(time_vector_pupil_per_move))]
    Xsens_position_no_level_CoM_corrected = [np.zeros(np.shape(Xsens_position_per_move[i])) for i in range(len(time_vector_pupil_per_move))]
    for j in range(len(time_vector_pupil_per_move)):
        start_time = time_vector_pupil_per_move[j][0]
        end_time = time_vector_pupil_per_move[j][-1]

        ToF_imove = end_time - start_time
        airborn_time = time_vector_pupil_per_move[j] - time_vector_pupil_per_move[j][0]

        CoM_initial_position = np.array([0, 0, hip_height]) - np.array([0, 0, Xsens_CoM_no_level[j][0, 2]])
        CoM_final_position = np.array([0, 0, hip_height]) - np.array([0, 0, Xsens_CoM_no_level[j][-1, 2]])
        CoM_initial_velocity = (CoM_final_position - CoM_initial_position - 0.5*-9.81*ToF_imove**2) / ToF_imove

        CoM_trajectory[j] = np.zeros((len(time_vector_pupil_per_move[j]), 3))
        CoM_trajectory[j][:, 2] = CoM_initial_position[2] + CoM_initial_velocity[2]*airborn_time + 0.5*-9.81*airborn_time**2

        hip_trajectory_imove = CoM_trajectory[j] - Xsens_CoM_no_level[j]

        for i in range(np.shape(Xsens_position_no_level_CoM_corrected[j])[0]):
            for k in range(num_joints):
                Xsens_position_no_level_CoM_corrected[j][i, 3*k:3*(k+1)] = Xsens_position_no_level[j][i,3*k:3*(k+1)] + hip_trajectory_imove[i, :]

    return Xsens_position_no_level_CoM_corrected, CoM_trajectory






