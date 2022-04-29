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


def CoM_transfo():

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
























