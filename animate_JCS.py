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
from gaze_position_gymnasium import get_gaze_position_from_intersection


def animate(Xsens_orientation, Xsens_position, Xsens_global_JCS_orientations, CoM_trajectory, elevation, azimuth, eye_position_height, eye_position_depth, links, num_joints, output_file_name, max_frame=0):

    # Make sure eyes are at the right place

    def Xsens_quat_to_orientation(Xsens_orientation, Xsens_position, Xsens_global_JCS_orientations, elevation, azimuth, eye_position_height, eye_position_depth, i_line):
        Xsens_head_position_calculated = np.zeros((6, ))
        Quat_normalized = Xsens_orientation[4*i_line:4*(i_line+1)] / np.linalg.norm(Xsens_orientation[4*i_line:4*(i_line+1)])
        Quat = biorbd.Quaternion(Quat_normalized[0], Quat_normalized[1], Quat_normalized[2], Quat_normalized[3])
        RotMat = biorbd.Quaternion.toMatrix(Quat).to_array()
        # # embed()
        # # print("np.all(Initial_RotMat == np.eye(3)) : ", np.all(Initial_RotMat == np.eye(3)))
        # # print("Initial_RotMat : ", Initial_RotMat)
        # # if i_line == 1 or i_line == 4:             # Initial_Quat = biorbd.Quaternion(1, 0, 0, 0)
        #     # Xsens_head_position_calculated[i_line][:3] = Xsens_position[3*i_line:3*(i_line+1)]
        #     # Xsens_head_position_calculated[i_line][3:] = RotMat @ np.array([0, 0, 0.1]) + Xsens_position[3*i_line:3*(i_line+1)]
        # # else:
        # Initial_Quat_normalized = Xsens_global_JCS_orientations[4 * i_line:4 * (i_line + 1)] / np.linalg.norm(Xsens_global_JCS_orientations[4 * i_line:4 * (i_line + 1)])
        # Initial_Quat = biorbd.Quaternion(Initial_Quat_normalized[0], Initial_Quat_normalized[1],Initial_Quat_normalized[2], Initial_Quat_normalized[3])
        # Initial_RotMat = biorbd.Quaternion.toMatrix(Initial_Quat).to_array()
        Xsens_head_position_calculated[:3] = Xsens_position[3 * i_line:3 * (i_line + 1)]
        Xsens_head_position_calculated[3:] = RotMat @ np.array([0, 0, 0.1]) + Xsens_position[3 * i_line:3 * (i_line + 1)]
        # RotMat @ np.linalg.inv(Initial_RotMat) @ np.array([]) np.linalg.inv(Initial_RotMat) @
        eye_position = RotMat @ np.array([eye_position_depth, 0, eye_position_height]) + Xsens_position[3 * i_line:3 * (i_line + 1)]
        gaze_rotMat = biorbd.Rotation_fromEulerAngles(np.array([azimuth, elevation]), 'zy').to_array()
        gaze_orientation = gaze_rotMat @ RotMat @ np.array([10, 0, 0]) + eye_position

        return Xsens_head_position_calculated, eye_position, gaze_orientation


    def compute_eye_related_positions(Xsens_orientation, Xsens_position, Xsens_global_JCS_orientations, elevation, azimuth, eye_position_height, eye_position_depth):

        Xsens_head_position_calculated = np.zeros((np.shape(Xsens_position)[0], 6))
        eye_position = np.zeros((np.shape(Xsens_position)[0], 3))
        gaze_orientation = np.zeros((np.shape(Xsens_position)[0], 3))
        gaze_intersection = np.zeros((np.shape(Xsens_position)[0], 3))

        for i_time in range(len(Xsens_orientation)):
            Xsens_head_position_calculated[i_time, :], eye_position[i_time, :], gaze_orientation[i_time, :] = Xsens_quat_to_orientation(Xsens_orientation[i_time, :], Xsens_position[i_time, :], Xsens_global_JCS_orientations, elevation[i_time], azimuth[i_time], eye_position_height, eye_position_depth, 6)
            gaze_intersection[i_time, :], _ = get_gaze_position_from_intersection(eye_position[i_time, :], gaze_orientation[i_time, :])

        return Xsens_head_position_calculated, eye_position, gaze_orientation, gaze_intersection


    def update(i, CoM_trajectory, Xsens_position, Xsens_head_position_calculated, eye_position, gaze_orientation, gaze_intersection, lines, CoM_point, line_eye_orientation, eyes_point, intersection_point, links):

        CoM_point[0][0].set_data(np.array([CoM_trajectory[i, 0]]), np.array([CoM_trajectory[i, 1]]))
        CoM_point[0][0].set_3d_properties(np.array([CoM_trajectory[i, 2]]))

        for i_line, line in enumerate(lines):
            line[0].set_data(np.array([Xsens_position[i, 3 * links[i_line, 0]], Xsens_position[i][3 * links[i_line, 1]] ]), np.array([Xsens_position[i, 3 * links[i_line, 0] + 1], Xsens_position[i, 3 * links[i_line, 1] + 1] ]))
            line[0].set_3d_properties(np.array([Xsens_position[i, 3 * links[i_line, 0] + 2], Xsens_position[i, 3 * links[i_line, 1] + 2] ]))

        line[0].set_data(np.array([Xsens_head_position_calculated[i, 0], Xsens_head_position_calculated[i, 3]]), np.array([Xsens_head_position_calculated[i, 1], Xsens_head_position_calculated[i, 4]]))
        line[0].set_3d_properties(np.array([Xsens_head_position_calculated[i, 2], Xsens_head_position_calculated[i, 5]]))
        eyes_point[0][0].set_data(np.array([eye_position[i, 0]]), np.array([eye_position[i, 1]]))
        eyes_point[0][0].set_3d_properties(np.array([eye_position[i, 2]]))
        line_eye_orientation[0][0].set_data(np.array([eye_position[i, 0], gaze_orientation[i, 0]]), np.array([eye_position[i, 1], gaze_orientation[i, 1]]))
        line_eye_orientation[0][0].set_3d_properties(np.array([eye_position[i, 2], gaze_orientation[i, 2]]))

        if gaze_intersection is not None:
            intersection_point[0][0].set_data(np.array([gaze_intersection[i, 0]]), np.array([gaze_intersection[i, 1]]))
            intersection_point[0][0].set_3d_properties(np.array([gaze_intersection[i, 2]]))

        return


    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_box_aspect([1, 1, 1])

    # Front right, to front left (bottom)
    plt.plot(np.array([7.193, 7.360]),
             np.array([-121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([0, 0]),
             '-k')
    # Front right, to back right (bottom)
    plt.plot(np.array([-8.881, 7.193]),
             np.array([-121 * 0.0254 / 2, -121 * 0.0254 / 2]),
             np.array([0, 0]),
             '-k')
    # Front left, to back left (bottom)
    plt.plot(np.array([-8.881, 7.360]),
             np.array([121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([0, 0]),
             '-k')
    # Back right, to back left (bottom)
    plt.plot(np.array([-8.881, -8.881]),
             np.array([-121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([0, 0]),
             '-k')

    # Front right, to front left (ceiling)
    plt.plot(np.array([7.193, 7.360]),
             np.array([-121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
             '-k')
    # Front right, to back right (ceiling)
    plt.plot(np.array([-8.881, 7.193]),
             np.array([-121 * 0.0254 / 2, -121 * 0.0254 / 2]),
             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
             '-k')
    # Front left, to back left (ceiling)
    plt.plot(np.array([-8.881, 7.360]),
             np.array([121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
             '-k')
    # Back right, to back left (ceiling)
    plt.plot(np.array([-8.881, -8.881]),
             np.array([-121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
             '-k')

    # Front right bottom, to front right ceiling
    plt.plot(np.array([7.193, 7.193]),
             np.array([-121 * 0.0254 / 2, -121 * 0.0254 / 2]),
             np.array([0, 9.4620 - 1.2192]),
             '-k')
    # Front left bottom, to front left ceiling
    plt.plot(np.array([7.360, 7.360]),
             np.array([121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([0, 9.4620 - 1.2192]),
             '-k')
    # Back right bottom, to back right ceiling
    plt.plot(np.array([-8.881, -8.881]),
             np.array([-121 * 0.0254 / 2, -121 * 0.0254 / 2]),
             np.array([0, 9.4620 - 1.2192]),
             '-k')
    # Back left bottom, to back left ceiling
    plt.plot(np.array([-8.881, -8.881]),
             np.array([121 * 0.0254 / 2, 121 * 0.0254 / 2]),
             np.array([0, 9.4620 - 1.2192]),
             '-k')


    # Trampoline
    X, Y = np.meshgrid([-7*0.3048, 7*0.3048], [-3.5*0.3048, 3.5*0.3048])
    Z =  np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, color='k', alpha=0.2)


    CoM_point = [ax.plot(0, 0, 0, '.r')]
    eyes_point = [ax.plot(0, 0, 0, '.c')]
    intersection_point = [ax.plot(0, 0, 0, '.c', markersize=10)]
    lines = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-k') for _ in range(len(links))]
    line_eye_orientation = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), '-b')]

    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    if max_frame == 0:
        frame_range = range(len(Xsens_position))
    else:
        frame_range = range(max_frame)

    Xsens_head_position_calculated, eye_position, gaze_orientation, gaze_intersection = compute_eye_related_positions(Xsens_orientation, Xsens_position, Xsens_global_JCS_orientations, elevation, azimuth, eye_position_height, eye_position_depth)




    anim = animation.FuncAnimation(fig, update, frames=frame_range, fargs=(CoM_trajectory,
                                                                           Xsens_position,
                                                                           Xsens_head_position_calculated,
                                                                           eye_position,
                                                                           gaze_orientation,
                                                                           gaze_intersection,
                                                                           lines,
                                                                           CoM_point,
                                                                           line_eye_orientation,
                                                                           eyes_point,
                                                                           intersection_point,
                                                                           links), blit=False)

    anim.save(output_file_name, fps=60, extra_args=['-vcodec', 'libx264'])
    # plt.show()

    return






