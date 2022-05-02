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
from casadi import *

def sync_jump(Xsens_sensorFreeAcceleration, start_of_jump_index, end_of_jump_index, start_of_move_index, end_of_move_index, FLAG_SYNCHRO_PLOTS, csv_imu, Xsens_ms):

    time_vector_pupil = (csv_imu[:, 0] - csv_imu[0, 0]) / 1e9
    time_vector_xsens = (Xsens_ms - Xsens_ms[0]) / 1000

    moving_average_window_size = 3
    Xsens_sensorFreeAcceleration_averaged = np.zeros((len(Xsens_sensorFreeAcceleration), 3))
    for j in range(3):
        for i in range(len(Xsens_sensorFreeAcceleration)):
            if i < moving_average_window_size:
                Xsens_sensorFreeAcceleration_averaged[i, j] = np.mean(Xsens_sensorFreeAcceleration[:2 * i + 1, j + 6])
            elif i > (len(Xsens_sensorFreeAcceleration) - moving_average_window_size - 1):
                Xsens_sensorFreeAcceleration_averaged[i, j] = np.mean(Xsens_sensorFreeAcceleration[-2 * (len(Xsens_sensorFreeAcceleration) - i) + 1:, j + 6])
            else:
                Xsens_sensorFreeAcceleration_averaged[i, j] = np.mean(Xsens_sensorFreeAcceleration[i - moving_average_window_size : i + moving_average_window_size + 1, j + 6])

    # moving_average_window_size_smoothest = 30
    # Xsens_sensorFreeAcceleration_averaged_smoothest = np.zeros((len(Xsens_sensorFreeAcceleration), 3))
    # for j in range(3):
    #     for i in range(len(Xsens_sensorFreeAcceleration)):
    #         if i < moving_average_window_size:
    #             Xsens_sensorFreeAcceleration_averaged_smoothest[i, j] = np.mean(Xsens_sensorFreeAcceleration[:2 * i + 1, j + 6])
    #         elif i > (len(Xsens_sensorFreeAcceleration) - moving_average_window_size - 1):
    #             Xsens_sensorFreeAcceleration_averaged_smoothest[i, j] = np.mean(Xsens_sensorFreeAcceleration[-2 * (len(Xsens_sensorFreeAcceleration) - i) + 1:, j + 6])
    #         else:
    #             Xsens_sensorFreeAcceleration_averaged_smoothest[i, j] = np.mean(Xsens_sensorFreeAcceleration[i - moving_average_window_size : i + moving_average_window_size + 1, j + 6])

    if FLAG_SYNCHRO_PLOTS:
        plt.figure()
        plt.plot(Xsens_sensorFreeAcceleration_averaged, '-', label=f"averaged {moving_average_window_size}")
        # plt.plot(Xsens_sensorFreeAcceleration_averaged_smoothest, '-', linewidth=3, label=f"averaged {moving_average_window_size_smoothest}")
        plt.plot(Xsens_sensorFreeAcceleration[:, 6:9], ':', label="raw")
        plt.legend()
        plt.title('Xsens')
        plt.show()

    Xsens_sensorFreeAcceleration_averaged_norm = np.linalg.norm(Xsens_sensorFreeAcceleration_averaged, axis=1)
    # Xsens_sensorFreeAcceleration_averaged_norm_smoothest = np.linalg.norm(Xsens_sensorFreeAcceleration_averaged_smoothest, axis=1)

    if FLAG_SYNCHRO_PLOTS:

        peaks_max, _ = signal.find_peaks(Xsens_sensorFreeAcceleration_averaged_norm, prominence=(10, None))
        peaks_min, _ = signal.find_peaks(-Xsens_sensorFreeAcceleration_averaged_norm, prominence=(2, None))
        # peaks_total = np.sort(np.hstack((peaks_min, peaks_max)))

        plt.figure()
        plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, '-k', alpha=0.5, label=f"averaged {moving_average_window_size}")
        # plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm_smoothest, '-k', linewidth=5, alpha=0.5, label=f"averaged {moving_average_window_size_smoothest}")
        plt.plot(time_vector_xsens[peaks_max], Xsens_sensorFreeAcceleration_averaged_norm[peaks_max], 'xr')
        plt.plot(time_vector_xsens[peaks_min], Xsens_sensorFreeAcceleration_averaged_norm[peaks_min], 'xm')

        candidate_start = []
        candidate_end = []
        for i in range(len(peaks_min) - 1):
            xsens_std = np.std(Xsens_sensorFreeAcceleration_averaged_norm[peaks_min[i] + 5: peaks_min[i+1] - 5])
            if xsens_std < 1 and np.abs(time_vector_xsens[peaks_min[i]] - time_vector_xsens[peaks_min[i+1]]) > 0.5:
                if i == 0:
                    plt.plot(time_vector_xsens[peaks_min[i] : peaks_min[i+1]],
                             Xsens_sensorFreeAcceleration_averaged_norm[peaks_min[i] : peaks_min[i+1]],
                             '-k', label="potential jump")
                else:
                    plt.plot(time_vector_xsens[peaks_min[i] : peaks_min[i+1]],
                             Xsens_sensorFreeAcceleration_averaged_norm[peaks_min[i] : peaks_min[i+1]],
                             '-k')
                candidate_start += [peaks_min[i]]
                candidate_end += [peaks_min[i+1]]


        diff_time = 10000
        time_offset = 0
        len_diff = len(candidate_start) - len(start_of_jump_index)
        for i in range(len_diff):  # considere qu'il y a toujours plus de candidats que de vrai sauts
            candidate_start_this_time = candidate_start[i: -(len_diff - i)]
            candidate_end_this_time = candidate_end[i: -(len_diff - i)]

            time_diff = SX.sym("time_diff", 1)
            f = sum1(((time_vector_pupil[start_of_jump_index.astype(int)] - time_diff) - np.reshape(
                time_vector_xsens[candidate_start_this_time], len(candidate_start_this_time))) ** 2)
            nlp = {'x': time_diff, 'f': f}
            MySolver = "ipopt"
            solver = nlpsol("solver", MySolver, nlp)
            sol = solver()

            if sol["f"] < diff_time:
                diff_time = np.array([sol["f"]])[0][0][0]
                time_offset = np.array([sol["x"]])[0][0][0]
                # xsens_start_index = candidate_start_this_time
                # xsens_end_index = candidate_end_this_time

        time_vector_pupil_offset = time_vector_pupil - time_offset
        xsens_start_of_jump_index = np.zeros((len(start_of_jump_index)))
        xsens_end_of_jump_index = np.zeros((len(start_of_jump_index)))
        for i in range(len(start_of_jump_index)):
            xsens_start_of_jump_index[i] = np.argmin(np.abs(
                time_vector_pupil_offset[int(start_of_jump_index[i])] - time_vector_xsens))
            xsens_end_of_jump_index[i] = np.argmin(np.abs(time_vector_pupil_offset[int(end_of_jump_index[i])] - time_vector_xsens))

        xsens_start_of_move_index = np.zeros((len(start_of_move_index)))
        xsens_end_of_move_index = np.zeros((len(start_of_move_index)))
        for i in range(len(start_of_move_index)):
            xsens_start_of_move_index[i] = np.argmin(np.abs(
                time_vector_pupil_offset[int(start_of_move_index[i])] - time_vector_xsens))
            xsens_end_of_move_index[i] = np.argmin(np.abs(time_vector_pupil_offset[int(end_of_move_index[i])] - time_vector_xsens))


        for i in range(len(start_of_jump_index)):
            if i == 0:
                plt.plot(np.ones((2, )) * time_vector_pupil[int(start_of_jump_index[i])] - time_offset,
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-c', label='Start of jump Pupil')
                plt.plot(np.ones((2,)) * time_vector_pupil[int(end_of_jump_index[i])] - time_offset,
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-c', label='End of jump Pupil')
                plt.plot(np.ones((2, )) * time_vector_xsens[int(xsens_start_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-b', label='Start of jump Xsens')
                plt.plot(np.ones((2,)) * time_vector_xsens[int(xsens_end_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-b', label='End of jump Xsens')
            else:
                plt.plot(np.ones((2,)) * time_vector_pupil[int(start_of_jump_index[i])] - time_offset,
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-c')
                plt.plot(np.ones((2,)) * time_vector_pupil[int(end_of_jump_index[i])] - time_offset,
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-c')
                plt.plot(np.ones((2,)) * time_vector_xsens[int(xsens_start_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-b')
                plt.plot(np.ones((2,)) * time_vector_xsens[int(xsens_end_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-b')

        for i in range(len(start_of_move_index)):
            if i == 0:
                plt.plot(np.ones((2, )) * time_vector_xsens[int(xsens_start_of_move_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-r', label='Start of move Xsens')
                plt.plot(np.ones((2,)) * time_vector_xsens[int(xsens_end_of_move_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-r', label='End of move Xsens')
            else:
                plt.plot(np.ones((2, )) * time_vector_xsens[int(xsens_start_of_move_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-r')
                plt.plot(np.ones((2,)) * time_vector_xsens[int(xsens_end_of_move_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-r')


        plt.legend()
        plt.title('Xsens')
        plt.show()

    return xsens_start_of_jump_index, xsens_end_of_jump_index, xsens_start_of_move_index, xsens_end_of_move_index, time_vector_xsens, time_vector_pupil_offset






