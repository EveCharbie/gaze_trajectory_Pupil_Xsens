
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sympy


def get_gaze_position_from_intersection(vector_origin, vector_orientation, planes_points, planes_normal_vectors):
    def intersection_plane_vector(vector_origin, vector_orientation, planes_points, planes_normal_vectors, epsilon=1e-6):
        """
        p0, p1: Define the line.
        p_co, p_no: define the plane:
            p_co Is a point on the plane (plane coordinate).
            p_no Is a normal vector defining the plane direction;
                 (does not need to be normalized).

        Return a Vector or None (when the intersection can't be found).
        from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
        """

        line_orientation = vector_origin + 10 * vector_orientation - vector_origin
        dot = planes_normal_vectors * line_orientation

        if abs(dot) > epsilon:
            # The factor of the point between vector_origin -> vector_origin + 10 * vector_orientation (0 - 1)
            # if 'fac' is between (0 - 1) the point intersects with the segment.
            # Otherwise:
            #  < 0.0: behind vector_origin.
            #  > 1.0: infront of vector_origin + 10 * vector_orientation.
            in_plane_vector = vector_origin - planes_points
            factor = - planes_normal_vectors * in_plane_vector / dot
            line_orientation = line_orientation * factor
            return vector_origin + line_orientation
        return None

    intersection = []
    for i in range(len(planes_points)):
        intersection += [intersection_plane_vector(vector_origin, vector_orientation, planes_points[i], planes_normal_vectors[i], epsilon=1e-6)]

        if intersection[i] != None:
            if intersection * vector_orientation > 0:
                intersection_index[i] = 1
            else:
                intersection_index[i] = 0
        else:
            intersection_index[i] = 0

    if intersection_index.sum() != 1:
        print('Probleme !')
    else:
        gaze_position = intersection[np.where(intersection_index == 1)]

    if intersection_index > 4:
        gaze_outside_side_bounds = True
    else:
        gaze_outside_side_bounds = False

    return gaze_position, gaze_outside_side_bounds, intersection_index


def unwrap_gaze_position(gaze_position):
              # Wall front
# Bound left  # Trampoline  # Bound right
              # Wall back
              # Ceiling

    if intersection_index[0] == 1: # trampoline
        gaze_position_x_y = gaze_position[:2]
    elif intersection_index[1] == 1: # wall front
        # wall front is not normal to the side bounds
        wall_front_vector = np.array([121 * 0.0254 / 2, 7.193, 0]) - np.array([-121 * 0.0254 / 2, 7.360, 0])
        gaze_position_2_norm = gaze_position[2]
        y_unknown = np.sqrt(gaze_position_2_norm ** 2 / (wall_front_vector[1]**2 / wall_front_vector[0]**2 + 1))
        x_unknown = -wall_front_vector[1] / wall_front_vector[0] * y_unknown
        gaze_position_x_y = (np.array([gaze_position[0], gaze_position[1]]) + np.array([x_unknown, y_unknown])).tolist()
    elif intersection_index[2] == 1: # ceiling
        gaze_position_x_y = [gaze_position[0], gaze_position[1] + 9.462 + 2*8.881]
    elif intersection_index[2] == 1:  # wall back
        gaze_position_x_y = [gaze_position[0], gaze_position[1] - gaze_position[2]]
    elif intersection_index[2] == 1:  # bound right
        gaze_position_x_y = [gaze_position[0] + gaze_position[2], gaze_position[1]]
    elif intersection_index[2] == 1:  # bound left
        gaze_position_x_y = [gaze_position[0] - gaze_position[2], gaze_position[1]]

    return gaze_position_x_y

################### Attention le plafond est 'la heuteur de la toile' trop haut
# ceiling height = 9.462 -
# zero is positioned at the center of the trampoline
planes_points = [[121 * 0.0254 / 2, 7.193, 0], # trampoline
                 [121 * 0.0254 / 2, 7.193, 0], # wall front
                 [121 * 0.0254 / 2, 7.193, 9.462], # ceiling
                 [121 * 0.0254 / 2, -8.881, 0], # wall back
                 [121 * 0.0254 / 2, 7.193, 0], # bound right
                 [-121 * 0.0254 / 2, 7.360, 0], # bound left
                 ]

planes_plots_info = []

x = np.linspace(-121 * 0.0254 / 2, 121 * 0.0254 / 2, 100)
y = np.linspace(-8.881, 7.360, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros((100, 100))
planes_plots_info.append([X, Y, Z]) # trampoline

x = np.linspace(-121 * 0.0254 / 2, 121 * 0.0254 / 2, 100)
z = np.linspace(0, 9.462, 100)
X, Z = np.meshgrid(x, z)
Y = ((7.193 - 7.360) / (2 * 121 * 0.0254 / 2)) * X + 7.360 - ((7.193 - 7.360) / (2 * 121 * 0.0254 / 2)) * (-121 * 0.0254 / 2)
planes_plots_info.append([X, Y, Z]) # wall front

x = np.linspace(-121 * 0.0254 / 2, 121 * 0.0254 / 2, 100)
y = np.linspace(-8.881, 7.360, 100)
X, Y = np.meshgrid(x, y)
Z = np.ones((100, 100)) * 9.462
planes_plots_info.append([X, Y, Z]) # ceiling

x = np.linspace(-121 * 0.0254 / 2, 121 * 0.0254 / 2, 100)
z = np.linspace(0, 9.462, 100)
X, Z = np.meshgrid(x, z)
Y = np.ones((100, 100)) * -8.881
planes_plots_info.append([X, Y, Z]) # wall back

y = np.linspace(-8.881, 7.193, 100)
z = np.linspace(0, 9.462, 100)
Y, Z = np.meshgrid(y, z)
X = np.ones((100, 100)) * 121 * 0.0254 / 2
planes_plots_info.append([X, Y, Z]) # bound right

y = np.linspace(-8.881, 7.360, 100)
z = np.linspace(0, 9.462, 100)
Y, Z = np.meshgrid(y, z)
X = np.ones((100, 100)) * -121 * 0.0254 / 2
planes_plots_info.append([X, Y, Z]) # bound left


planes_vector = [ [0, 0, 1 ], # trampoline
                np.cross(np.array([121 * 0.0254 / 2, 7.193, 0]) - np.array([-121 * 0.0254 / 2, 7.360, 0]), np.array([0, 0, 1])).tolist(), # wall front
                [ 0, 0, -1 ], # ceiling
                [ 0, 1, 0 ], # wall back
                [ -1, 0, 0 ], # bound right
                [ 1, 0, 0 ], # bound left
                ]



for j in range(len(xsens_data)):
    vector_origin, vector_orientation = get_vector_from_Xsens() # [0,0,0], [1,1,1]
    # vector_orientation should be unit vector
    gaze_position, gaze_outside_side_bounds, intersection_index = get_gaze_position_from_intersection(planes, vector_origin, vector_orientation)

    # see where the intersection is
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(planes_points)):
        ax.plot_surface(planes_plots_info[i][0], planes_plots_info[i][1], planes_plots_info[i][2])
    ax.set_xlabel('gauche/droite')
    ax.set_ylabel('derriere/devant')
    ax.set_zlabel('plancher/plafond')
    plt.show()


























