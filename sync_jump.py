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

def sync_jump(Xsens_sensorFreeAcceleration, start_of_jump_index, end_of_jump_index, FLAG_SYNCHRO_PLOTS, csv_imu, Xsens_ms):

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

    if FLAG_SYNCHRO_PLOTS:
        plt.figure()
        plt.plot(Xsens_sensorFreeAcceleration_averaged, '-', label="averaged")
        plt.plot(Xsens_sensorFreeAcceleration[:, 6:9], ':', label="raw")
        plt.legend()
        plt.title('Xsens')
        plt.show()

    Xsens_sensorFreeAcceleration_averaged_norm = np.linalg.norm(Xsens_sensorFreeAcceleration_averaged, axis=1)

    if FLAG_SYNCHRO_PLOTS:
        embed()

        peaks_max, _ = signal.find_peaks(Xsens_sensorFreeAcceleration_averaged_norm, prominence=(10, None))
        peaks_min, _ = signal.find_peaks(-Xsens_sensorFreeAcceleration_averaged_norm, prominence=(2, None))
        # peaks_total = np.sort(np.hstack((peaks_min, peaks_max)))

        plt.figure()
        plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, '-k', alpha=0.5, label="averaged")
        plt.plot(time_vector_xsens[peaks_max], Xsens_sensorFreeAcceleration_averaged_norm[peaks_max], 'xr')
        plt.plot(time_vector_xsens[peaks_min], Xsens_sensorFreeAcceleration_averaged_norm[peaks_min], 'xm')

        candidate_start = []
        candidate_end = []
        for i in range(len(peaks_min) - 1):
            xsens_std = np.std(Xsens_sensorFreeAcceleration_averaged_norm[peaks_min[i] + 5: peaks_min[i+1] - 5])
            if xsens_std < 1 and np.abs(time_vector_xsens[peaks_min[i]] - time_vector_xsens[peaks_min[i+1]]) > 0.5:
                plt.plot(time_vector_xsens[peaks_min[i] : peaks_min[i+1]],
                         Xsens_sensorFreeAcceleration_averaged_norm[peaks_min[i] : peaks_min[i+1]],
                         '-k', label="potential jump")
                candidate_start += [peaks_min[i]]
                candidate_end += [peaks_min[i+1]]

        for i in range(len(start_of_jump_index)):
            if i == 0:
                plt.plot(np.ones((2, )) * time_vector_pupil[int(start_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-k', label='Start of jump')
                plt.plot(np.ones((2,)) * time_vector_pupil[int(end_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-k', label='End of jump')
            else:
                plt.plot(np.ones((2,)) * time_vector_pupil[int(start_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-k')
                plt.plot(np.ones((2,)) * time_vector_pupil[int(end_of_jump_index[i])],
                         np.array([np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                                   np.max(Xsens_sensorFreeAcceleration_averaged_norm)]), '-k')

        plt.legend()
        plt.title('Xsens')
        plt.show()


    if len(candidate_start) == len(start_of_jump_index):
        time_diff = time_vector_pupil[int(start_of_jump_index[0])] - time_vector_xsens[candidate_start[0]]
    else:
        total_diff = 1000
        for i in range(len(candidate_start) - len(start_of_jump_index)):
            start_of_jump_index_chosen = start_of_jump_index[i : i + len(candidate_start)]
            time_diff = time_vector_pupil[int(start_of_jump_index[i])] - time_vector_xsens[candidate_start[0]]
            for j in range(len(candidate_start)):
            time_sync_error = np.sum(time_vector_pupil[int(start_of_jump_index[0])] - time_vector_xsens[candidate_start[0]])


    if FLAG_COM_PLOTS:
        # labels_CoM = ["X", "Y", "Z", "vitesse X", "vitesse Y", "vitesse Z", "acc X", "acc Y", "acc Z"]
        # plt.figure()
        # for i in range(3):
        #     plt.plot(Xsens_centerOfMass[:, i], label=f'{labels_CoM[i]}')
        # plt.legend()
        # plt.show()
        #
        # peaks_max, _ = signal.find_peaks(Xsens_centerOfMass[:, 5], prominence=(0.1, None))
        # peaks_min, _ = signal.find_peaks(-Xsens_centerOfMass[:, 5], prominence=(0.1, None))
        #
        # peaks_total = np.sort(np.hstack((peaks_min, peaks_max)))
        #
        # plt.figure()
        # for i in range(3, 6):
        #     plt.plot(time_vector_xsens, Xsens_centerOfMass[:, i], label=f'{labels_CoM[i]}')
        #     if i == 5:
        #         plt.plot(time_vector_xsens[peaks_max], Xsens_centerOfMass[peaks_max, 5], 'xg')
        #         plt.plot(time_vector_xsens[peaks_min], Xsens_centerOfMass[peaks_min, 5], 'xg')
        #         for j in range(len(peaks_total) - 1):
        #             x_linregress = np.reshape(time_vector_xsens[peaks_total[j]:peaks_total[j + 1]],
        #                                       (len(time_vector_xsens[peaks_total[j]:peaks_total[j + 1]]),))
        #             slope, intercept, _, _, _ = scipy.stats.linregress(x_linregress,
        #                                                                Xsens_centerOfMass[
        #                                                                peaks_total[j]: peaks_total[j + 1], 5])
        #             plt.plot(x_linregress,
        #                      intercept + slope * x_linregress,
        #                      '--k', alpha=0.5)
        #             print("slope : ", slope)
        #
        # plt.legend()
        # plt.show()

        plt.figure()
        plt.plot(time_vector_xsens, Xsens_centerOfMass[:, 5], label='acceleration CoM Z')
        plt.plot(time_vector_xsens[peaks_max], Xsens_centerOfMass[peaks_max, 5], 'xg')
        plt.plot(time_vector_xsens[peaks_min], Xsens_centerOfMass[peaks_min, 5], 'xg')
        for j in range(len(peaks_total) - 1):
            x_linregress = np.reshape(time_vector_xsens[peaks_total[j]:peaks_total[j + 1]],
                                      (len(time_vector_xsens[peaks_total[j]:peaks_total[j + 1]]),))
            slope, intercept, _, _, _ = scipy.stats.linregress(x_linregress,
                                                               Xsens_centerOfMass[peaks_total[j]: peaks_total[j + 1],
                                                               5])
            plt.plot(x_linregress,
                     intercept + slope * x_linregress,
                     '--k', alpha=0.5)
        plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, '-r',
                 label="norm acceleration tete IMU")

        plt.legend()
        plt.show()

        plt.figure()
        for i in range(6, 9):
            plt.plot(Xsens_centerOfMass[:, i], label=f'{labels_CoM[i]}')
        plt.legend()
        plt.show()





    if FLAG_SYNCHRO_PLOTS:
        plt.figure()
        plt.plot(Xsens_sensorFreeAcceleration_averaged, '-', label="averaged")
        plt.plot(Xsens_sensorFreeAcceleration[:, 6:9], ':', label="raw")
        plt.legend()
        plt.title('Xsens')
        plt.show()


    time_vector_pupil_interp = np.arange(0, time_vector_pupil[-1], 1 / 200)
    time_vector_xsens_interp = np.arange(0, time_vector_xsens[-1], 1 / 200)
    pupil_interp = scipy.interpolate.interp1d(time_vector_pupil, csv_imu_averaged_norm)
    xsens_interp = scipy.interpolate.interp1d(np.reshape(time_vector_xsens, (len(time_vector_xsens),)),
                                              Xsens_sensorFreeAcceleration_averaged_norm)
    norm_accelaration_pupil = pupil_interp(time_vector_pupil_interp)
    norm_accelaration_xsens = xsens_interp(time_vector_xsens_interp)

    correlation_sum = np.array([])
    glide_shift = 800
    if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
        for i in range(-glide_shift, 0):
            current_correlation = np.mean(
                np.abs(np.correlate(norm_accelaration_xsens[:len(norm_accelaration_pupil) + i],
                                    norm_accelaration_pupil[-i:len(norm_accelaration_xsens)],
                                    mode='valid')))
            correlation_sum = np.hstack((correlation_sum, current_correlation))

        length_diff = len(norm_accelaration_xsens) - len(norm_accelaration_pupil)
        for i in range(0, length_diff):
            current_correlation = np.mean(
                np.abs(np.correlate(norm_accelaration_xsens[i: len(norm_accelaration_pupil) + i],
                                    norm_accelaration_pupil,
                                    mode='valid')))
            correlation_sum = np.hstack((correlation_sum, current_correlation))

        for i in range(length_diff, length_diff + glide_shift):
            current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_xsens[i:],
                                                              norm_accelaration_pupil[
                                                              :len(norm_accelaration_pupil) - i],
                                                              mode='valid')))
            correlation_sum = np.hstack((correlation_sum, current_correlation))

    else:
        for i in range(-glide_shift, 0):
            current_correlation = np.mean(
                np.abs(np.correlate(norm_accelaration_pupil[:len(norm_accelaration_xsens) + i],
                                    norm_accelaration_xsens[-i:len(norm_accelaration_pupil)],
                                    mode='valid')))
            correlation_sum = np.hstack((correlation_sum, current_correlation))

        length_diff = len(norm_accelaration_pupil) - len(norm_accelaration_xsens)
        for i in range(0, length_diff):
            current_correlation = np.mean(
                np.abs(np.correlate(norm_accelaration_pupil[i: len(norm_accelaration_xsens) + i],
                                    norm_accelaration_xsens,
                                    mode='valid')))
            correlation_sum = np.hstack((correlation_sum, current_correlation))

        for i in range(length_diff, length_diff + glide_shift):
            current_correlation = np.mean(np.abs(np.correlate(norm_accelaration_pupil[i:],
                                                              norm_accelaration_xsens[
                                                              : len(norm_accelaration_xsens) - i + length_diff],
                                                              mode='valid')))
            correlation_sum = np.hstack((correlation_sum, current_correlation))

    if FLAG_SYNCHRO_PLOTS:
        plt.figure()
        plt.plot(correlation_sum)
        plt.title('Correlation')
        plt.show()

    idx_max = np.argmax(correlation_sum) - glide_shift
    glide_shift_time = idx_max * 1 / 200

    if FLAG_SYNCHRO_PLOTS:
        plt.figure()
        if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
            plt.plot(np.arange(idx_max, idx_max + len(norm_accelaration_pupil)), norm_accelaration_pupil, '-r',
                     label='Pupil')
            plt.plot(norm_accelaration_xsens, '-b', label='Xsens')
        else:
            plt.plot(norm_accelaration_pupil, '-r', label='Pupil')
            plt.plot(np.arange(idx_max, idx_max + len(norm_accelaration_xsens)), norm_accelaration_xsens, '-b',
                     label='Xsens')
        plt.legend()
        plt.title('Synchro interp')
        plt.show()

        plt.figure()
        if len(norm_accelaration_xsens) > len(norm_accelaration_pupil):  # Xsens > Pupil
            plt.plot(time_vector_pupil + glide_shift_time, csv_imu_averaged_norm, '-r', label='Pupil')
            plt.plot(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, '-b', label='Xsens')
        else:
            plt.plot(time_vector_pupil, csv_imu_averaged_norm, '-r', label='Pupil')
            plt.plot(time_vector_xsens + glide_shift_time, Xsens_sensorFreeAcceleration_averaged_norm, '-b',
                     label='Xsens')
        plt.legend()
        plt.title('Synchro pas interp')
        plt.show()

    return






