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

def gaze_orientation():
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

    return






















