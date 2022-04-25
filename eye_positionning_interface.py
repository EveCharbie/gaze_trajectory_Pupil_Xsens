
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import json
import tkinter as tk
from tkinter import filedialog
import csv



def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, number_of_points_to_label, eye_height, eye_depth, small_face_image_clone_before, small_side_image_clone_before
    small_face_image_clone_before = np.zeros(np.shape(small_face_image), dtype=np.uint8)
    small_face_image_clone_before[:] = small_face_image[:]
    small_side_image_clone_before = np.zeros(np.shape(small_side_image), dtype=np.uint8)
    small_side_image_clone_before[:] = small_side_image[:]

    for i in range(9):
        mouse_click_position = (int(points_labels[label_keys[i]][0]), int(points_labels[label_keys[i]][1]))
        cv2.circle(small_face_image_clone_before, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
        cv2.putText(small_face_image_clone_before, point_label_names[i], (mouse_click_position[0]+3, mouse_click_position[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for i in range(9, 13):
        mouse_click_position = (int(points_labels[label_keys[i]][0]), int(points_labels[label_keys[i]][1]))
        cv2.circle(small_side_image_clone_before, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
        cv2.putText(small_side_image_clone_before, point_label_names[i], (mouse_click_position[0]+3, mouse_click_position[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if np.sum(points_labels[label_keys[0]]) != 0 and np.sum(points_labels[label_keys[1]]) != 0:
        line_position_Ankle = (int(points_labels[label_keys[0]][0]), int(points_labels[label_keys[0]][1]))
        line_position_Knee = (int(points_labels[label_keys[1]][0]), int(points_labels[label_keys[1]][1]))
        cv2.line(small_face_image_clone_before, line_position_Ankle, line_position_Knee, line_color, thickness=1)
    if np.sum(points_labels[label_keys[1]]) != 0 and np.sum(points_labels[label_keys[2]]) != 0:
        line_position_Knee = (int(points_labels[label_keys[1]][0]), int(points_labels[label_keys[1]][1]))
        line_position_HipRight = (int(points_labels[label_keys[2]][0]), int(points_labels[label_keys[2]][1]))
        cv2.line(small_face_image_clone_before, line_position_Knee, line_position_HipRight, line_color, thickness=1)
    if np.sum(points_labels[label_keys[2]]) != 0 and np.sum(points_labels[label_keys[3]]) != 0:
        line_position_HipRight = (int(points_labels[label_keys[2]][0]), int(points_labels[label_keys[2]][1]))
        line_position_HipLeft = (int(points_labels[label_keys[3]][0]), int(points_labels[label_keys[3]][1]))
        cv2.line(small_face_image_clone_before, line_position_HipRight, line_position_HipLeft, line_color, thickness=1)
    if np.sum(points_labels[label_keys[4]]) != 0 and np.sum(points_labels[label_keys[5]]) != 0:
        line_position_ShoulderRight = (int(points_labels[label_keys[4]][0]), int(points_labels[label_keys[4]][1]))
        line_position_ShoulderLeft = (int(points_labels[label_keys[5]][0]), int(points_labels[label_keys[5]][1]))
        cv2.line(small_face_image_clone_before, line_position_ShoulderRight, line_position_ShoulderLeft, line_color, thickness=1)
    if np.sum(points_labels[label_keys[2]]) != 0 and np.sum(points_labels[label_keys[3]]) != 0 and np.sum(points_labels[label_keys[4]]) != 0 and np.sum(points_labels[label_keys[5]]) != 0:
        line_position_Shoulder = (int((points_labels[label_keys[4]][0] + points_labels[label_keys[5]][0])/2), int((points_labels[label_keys[4]][1] + points_labels[label_keys[5]][1])/2))
        line_position_Hips = (int((points_labels[label_keys[2]][0] + points_labels[label_keys[3]][0])/2), int((points_labels[label_keys[2]][1] + points_labels[label_keys[3]][1])/2))
        cv2.line(small_face_image_clone_before, line_position_Hips, line_position_Shoulder, line_color, thickness=1)
    if np.sum(points_labels[label_keys[4]]) != 0 and np.sum(points_labels[label_keys[5]]) != 0 and np.sum(points_labels[label_keys[6]]) != 0:
        line_position_Shoulders = (int((points_labels[label_keys[4]][0] + points_labels[label_keys[5]][0])/2), int((points_labels[label_keys[4]][1] + points_labels[label_keys[5]][1])/2))
        line_position_Head = (int(points_labels[label_keys[6]][0]), int(points_labels[label_keys[6]][1]))
        cv2.line(small_face_image_clone_before, line_position_Shoulders, line_position_Head, line_color,thickness=1)
    if np.sum(points_labels[label_keys[7]]) != 0 and np.sum(points_labels[label_keys[8]]) != 0:
        line_position_C1 = (int(points_labels[label_keys[7]][0]), int(points_labels[label_keys[7]][1]))
        line_position_eye = (int(points_labels[label_keys[8]][0]), int(points_labels[label_keys[8]][1]))
        cv2.line(small_face_image_clone_before, line_position_C1, line_position_eye, line_color, thickness=3)

    if np.sum(points_labels[label_keys[9]]) != 0 and np.sum(points_labels[label_keys[10]]) != 0:
        line_position_Heel_side = (int(points_labels[label_keys[9]][0]), int(points_labels[label_keys[9]][1]))
        line_position_Toe_side = (int(points_labels[label_keys[10]][0]), int(points_labels[label_keys[10]][1]))
        cv2.line(small_side_image_clone_before, line_position_Heel_side, line_position_Toe_side, line_color, thickness=1)
    if np.sum(points_labels[label_keys[11]]) != 0 and np.sum(points_labels[label_keys[12]]) != 0:
        line_position_C1_side = (int(points_labels[label_keys[11]][0]), int(points_labels[label_keys[11]][1]))
        line_position_eye_side = (int(points_labels[label_keys[12]][0]), int(points_labels[label_keys[12]][1]))
        cv2.line(small_side_image_clone_before, line_position_C1_side, line_position_eye_side, line_color, thickness=3)

    cv2.imshow("Face", small_face_image_clone_before)
    cv2.imshow("Side", small_side_image_clone_before)

    points_active_number = 0
    for i in range(13):
        if np.sum(points_labels[label_keys[i]]) != 0:
            points_active_number += 1

    if points_active_number == 13:
        eye_height_pixel = np.sqrt((line_position_C1[0] - line_position_eye[0])**2 + (line_position_C1[1] - line_position_eye[1])**2)
        tibia_ratio = (knee_height - ankle_height) / np.sqrt((line_position_Ankle[0] - line_position_Knee[0])**2 + (line_position_Ankle[1] - line_position_Knee[1])**2)
        tigh_ratio = (hip_height - knee_height) / np.sqrt((line_position_Knee[0] - line_position_Hips[0]) ** 2 + (line_position_Knee[1] - line_position_Hips[1]) ** 2)
        trunk_ratio = (shoulder_height - hip_height) / np.sqrt((line_position_Shoulders[0] - line_position_Hips[0]) ** 2 + (line_position_Shoulders[1] - line_position_Hips[1]) ** 2)
        head_ratio = (body_height - shoulder_height) / np.sqrt((line_position_Head[0] - line_position_Shoulders[0]) ** 2 + (line_position_Head[1] - line_position_Shoulders[1]) ** 2)
        eye_height = np.mean(np.array([tibia_ratio, tigh_ratio, trunk_ratio, head_ratio])) * eye_height_pixel

        eye_depth_pixel = np.sqrt((line_position_C1_side[0] - line_position_eye_side[0])**2 + (line_position_C1_side[1] - line_position_eye_side[1])**2)
        foot_ratio = foot_length / np.sqrt((line_position_Heel_side[0] - line_position_Toe_side[0])**2 + (line_position_Heel_side[1] - line_position_Toe_side[1])**2)
        eye_depth = foot_ratio * eye_depth_pixel
    else:
        eye_height = 0
        eye_depth = 0
    return


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]] = np.array([x, y])
        draw_points_and_lines()
    return

def put_text():
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (53, 214)
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    cv2.putText(small_face_image_clone_before, f'Eye height: {eye_height}', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(small_side_image_clone_before, f'Eye depth: {eye_depth}', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Face", small_face_image_clone_before)
    cv2.imshow("Side", small_side_image_clone_before)
    return


def point_choice(*args):
    global current_click
    num_point = args[1]
    current_click = num_point
    draw_points_and_lines()
    return

# def save_bing_bong():
#     global display_image
#     display_image = False
#     return

# ############################### code beginning #######################################################################
global points_labels, current_click, point_label_names, display_image

# global image_clone, small_image, number_of_points_to_label, width_small, height_small, frame_counter, label_keys, points_labels, frames_clone
# global ratio_image, Image_name, borders_points, curent_AOI_label, csv_eye_tracking, point_label_names

circle_radius = 5
line_color = (1, 1, 1)
number_of_points_to_label = 13
circle_colors = sns.color_palette(palette="viridis", n_colors=number_of_points_to_label)
for i in range(number_of_points_to_label):
    col_0 = int(circle_colors[i][0] * 255)
    col_1 = int(circle_colors[i][1] * 255)
    col_2 = int(circle_colors[i][2] * 255)
    circle_colors[i] = (col_0, col_1, col_2)

face_ratio_image = 5
side_ratio_image = 3

subject_name = 'GuSe'
out_path = '/home/user/Documents/Programmation/rectangle-labelling/output/Results/'

if not os.path.exists(f'{out_path}/{subject_name}'):
    os.makedirs(f'{out_path}/{subject_name}')
if not os.path.exists(f'{out_path}/{subject_name}/{subject_name}_anthropo.csv') :
    f = open(f'{out_path}/{subject_name}/{subject_name}_anthropo.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Measure', '[m]'])
    writer.writerow(['body_height'])
    writer.writerow(['shoulder_height'])
    writer.writerow(['hip_height'])
    writer.writerow(['knee_height'])
    writer.writerow(['ankle_height'])
    writer.writerow(['foot_length'])
    writer.writerow(['hip_width'])
    writer.writerow(['shoulder_width'])
    writer.writerow(['elbow_span'])
    writer.writerow(['wrist_span'])
    writer.writerow(['arm_span'])
    writer.writerow(['eye_height'])
    writer.writerow(['eye_depth'])
    f.close()
    raise RuntimeError("Please provide anthropo measurements to continue")
else:
    csv_name = f'{out_path}/{subject_name}/{subject_name}_anthropo.csv'
    csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')
    body_height = float(csv_table[0][0][1])
    shoulder_height = float(csv_table[1][0][1])
    hip_height = float(csv_table[2][0][1])
    knee_height = float(csv_table[3][0][1])
    ankle_height = float(csv_table[4][0][1])
    foot_length = float(csv_table[5][0][1])
    hip_width = float(csv_table[6][0][1])
    shoulder_width = float(csv_table[7][0][1])
    elbow_span = float(csv_table[8][0][1])
    wrist_span = float(csv_table[9][0][1])
    arm_span = float(csv_table[10][0][1])

csv_name = "/home/user/Documents/Programmation/rectangle-labelling/Trials_name_mapping.csv"
csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')

for i in range(len(csv_table)):
    if subject_name == csv_table[i][0][0]:
        picture_face_path = '/home/user/Documents/Eye-tracking/Participants_pictures/' + csv_table[i][0][9]
        picture_side_path = '/home/user/Documents/Eye-tracking/Participants_pictures/' + csv_table[i][0][10]

point_label_names = ["Ankle", "Knee", "HipRight", "LeftHip", "ShoulderRight", "ShoulderLeft", "Top head", "C1", "Eye height", "Heel (side)", "Toe (side)", "C1 (side)", "Eye depth"]
points_labels = {"Ankle": np.zeros((2, )),
                "Knee": np.zeros((2, )),
                "HipRight": np.zeros((2, )),
                "HipLeft": np.zeros((2, )),
                "ShoulderRight": np.zeros((2, )),
                "ShoulderLeft": np.zeros((2, )),
                "Top head": np.zeros((2, )),
                "C1": np.zeros((2, )),
                "Eye height": np.zeros((2, )),
                "Heel (side)": np.zeros((2, )),
                "Toe (side)": np.zeros((2, )),
                "C1 (side)": np.zeros((2, )),
                "Eye depth": np.zeros((2, ))}

label_keys = [key for key in points_labels.keys()]
current_click = 0

def nothing(x):
    return

cv2.namedWindow("Face")
cv2.namedWindow("Side")

cv2.createButton("Ankle", point_choice, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Knee", point_choice, 1, cv2.QT_PUSH_BUTTON, 1)
cv2.createButton("HipRight", point_choice, 2, cv2.QT_PUSH_BUTTON, 2)
cv2.createButton("HipLeft", point_choice, 3, cv2.QT_PUSH_BUTTON, 3)
cv2.createButton("ShoulderRight", point_choice, 4, cv2.QT_PUSH_BUTTON, 4)
cv2.createButton("ShoulderLeft", point_choice, 5, cv2.QT_PUSH_BUTTON, 5)
cv2.createButton("Top head", point_choice, 6, cv2.QT_PUSH_BUTTON, 6)
cv2.createButton("C1", point_choice, 7, cv2.QT_PUSH_BUTTON, 7)
cv2.createButton("Eye height", point_choice, 8, cv2.QT_PUSH_BUTTON, 8)
cv2.createButton("Heel (side)", point_choice, 9, cv2.QT_PUSH_BUTTON, 9)
cv2.createButton("Toe (side)", point_choice, 10, cv2.QT_PUSH_BUTTON, 10)
cv2.createButton("C1 (side)", point_choice, 11, cv2.QT_PUSH_BUTTON, 11)
cv2.createButton("Eye depth", point_choice, 12, cv2.QT_PUSH_BUTTON, 12)
# cv2.createButton("Save", point_choice, 0, cv2.QT_PUSH_BUTTON, 0)

cv2.setMouseCallback("Face", mouse_click)
cv2.setMouseCallback("Side", mouse_click)

image_face = cv2.imread(picture_face_path)
image_side = cv2.imread(picture_side_path)
image_face_clone = image_face.copy()
image_side_clone = image_side.copy()
width_face, height_face, rgb_face = np.shape(image_face_clone)
width_side, height_side, rgb_side = np.shape(image_side_clone)
small_face_image = cv2.resize(image_face_clone, (int(round(height_face / face_ratio_image)), int(round(width_face / face_ratio_image))))
small_side_image = cv2.resize(image_side_clone, (int(round(height_side / side_ratio_image)), int(round(width_side / side_ratio_image))))
# width_small, height_small, rgb_small = np.shape(small_image)
cv2.imshow("Face", small_face_image)
cv2.imshow("Side", small_side_image)

display_image = True
while display_image == True:

    key = cv2.waitKey(0) & 0xFF
    draw_points_and_lines()

    if key == ord('x'):  # if `x` then quit
        put_text()
        display_image = False

cv2.destroyAllWindows()



rows = []
with open(f'{out_path}/{subject_name}/{subject_name}_anthropo.csv', 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
print(header)
print(rows)

f = open(f'{out_path}/{subject_name}/{subject_name}_anthropo.csv', 'w')
writer = csv.writer(f)
writer.writerow(header)
for i in range(len(rows)-2):
    writer.writerow(rows[i])
writer.writerow(['eye_height', f'{eye_height}'])
writer.writerow(['eye_depth', f'{eye_depth}'])
f.close()

