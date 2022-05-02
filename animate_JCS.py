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


def animate(Xsens_orientation, Xsens_position, CoM_trajectory, elevation, azimuth, eye_position_height, eye_position_depth, links, num_joints, output_file_name, max_frame=0):

    # Make sure eyes are at the right place
    # Plot walls
    # par plane_vector_crossing

    def Xsens_quat_to_orientation(Xsens_orientation, Xsens_position, elevation, azimuth, eye_position_height, eye_position_depth, i_line):
        Xsens_position_calculated = [np.zeros((6, )) for _ in range(num_joints)]
        Quat_normalized = Xsens_orientation[4*i_line:4*(i_line+1)] / np.linalg.norm(Xsens_orientation[4*i_line:4*(i_line+1)])
        Quat = biorbd.Quaternion(Quat_normalized[0], Quat_normalized[1], Quat_normalized[2], Quat_normalized[3])
        RotMat = biorbd.Quaternion.toMatrix(Quat).to_array()
        Xsens_position_calculated[i_line][:3] = Xsens_position[3*i_line:3*(i_line+1)]
        Xsens_position_calculated[i_line][3:] = RotMat @ np.array([0, 0, 0.1]) + Xsens_position[3*i_line:3*(i_line+1)]
        if i_line == 6:
            eye_position = RotMat @ np.array([eye_position_depth, 0, eye_position_height]) + Xsens_position[3 * i_line:3 * (i_line + 1)]
            gaze_rotMat = biorbd.Rotation_fromEulerAngles(np.array([azimuth, elevation]), 'xz').to_array()
            gaze_orientation = gaze_rotMat @ RotMat @ np.array([5, 0, 0]) + eye_position
        else:
            eye_position = np.array([0, 0, 0])
            gaze_orientation = np.array([0, 0, 0])
        return Xsens_position_calculated, eye_position, gaze_orientation


    def update(i, Xsens_orientation, Xsens_position, CoM_trajectory, elevation, azimuth, lines, CoM_point, line_orientation, line_eye_orientation, eyes_point, eye_position_height, eye_position_depth, links):

        CoM_point[0][0].set_data(np.array([CoM_trajectory[i, 0]]), np.array([CoM_trajectory[i, 1]]))
        CoM_point[0][0].set_3d_properties(np.array([CoM_trajectory[i, 2]]))

        for i_line, line in enumerate(lines):
            line[0].set_data(np.array([Xsens_position[i, 3 * links[i_line, 0]], Xsens_position[i][3 * links[i_line, 1]] ]), np.array([Xsens_position[i, 3 * links[i_line, 0] + 1], Xsens_position[i, 3 * links[i_line, 1] + 1] ]))
            line[0].set_3d_properties(np.array([Xsens_position[i, 3 * links[i_line, 0] + 2], Xsens_position[i, 3 * links[i_line, 1] + 2] ]))

        for i_line, line in enumerate(line_orientation[:7]): # enumerate(line_orientation):
            Xsens_position_calculated, eye_position, gaze_orientation = Xsens_quat_to_orientation(Xsens_orientation[i, :], Xsens_position[i, :], elevation[i], azimuth[i], eye_position_height, eye_position_depth, i_line)
            line[0].set_data(np.array([Xsens_position_calculated[i_line][0], Xsens_position_calculated[i_line][3]]), np.array([Xsens_position_calculated[i_line][1], Xsens_position_calculated[i_line][4]]))
            line[0].set_3d_properties(np.array([Xsens_position_calculated[i_line][2], Xsens_position_calculated[i_line][5]]))
            if i_line == 6:
                eyes_point[0][0].set_data(np.array([eye_position[0]]), np.array([eye_position[1]]))
                eyes_point[0][0].set_3d_properties(np.array([eye_position[2]]))
                line_eye_orientation[0][0].set_data(np.array([eye_position[0], gaze_orientation[0]]), np.array([eye_position[1], gaze_orientation[1]]))
                line_eye_orientation[0][0].set_3d_properties(np.array([eye_position[2], gaze_orientation[2]]))
        return

    for j in range(len(Xsens_position)):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_box_aspect([1, 1, 1])

        CoM_point = [ax.plot(0, 0, 0, '.r')]
        eyes_point = [ax.plot(0, 0, 0, '.c')]
        lines = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-k') for _ in range(len(links))]
        line_orientation = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-m') for _ in range(num_joints)]
        line_eye_orientation = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-b')]

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
            frame_range = range(len(Xsens_position[j]))
        else:
            frame_range = range(max_frame)

        anim = animation.FuncAnimation(fig, update, frames=frame_range, fargs=(Xsens_orientation[j],
                                                                               Xsens_position[j],
                                                                               CoM_trajectory[j],
                                                                               elevation[j],
                                                                               azimuth[j],
                                                                               lines,
                                                                               CoM_point,
                                                                               line_orientation,
                                                                               line_eye_orientation,
                                                                               eyes_point,
                                                                               eye_position_height,
                                                                               eye_position_depth,
                                                                               links), blit=False)

        anim.save(output_file_name, fps=60, extra_args=['-vcodec', 'libx264'])
        plt.show()

    return






