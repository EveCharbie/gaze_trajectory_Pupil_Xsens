% this code allows to create a map of the gaze trajectory from Xsens +
% eye-tracker measurements on trampolinists


clear all; close all; clc;

% file_dir = '/home/user/Documents/Eye-tracking/XsensData';
file_dir = '/home/user/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData';
% file_dir = '/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData';
file_name = 'Test_17032021-020.mvnx';

% mvnx = load_mvnx([file_dir, '/exports/' file_name])
mvnx = load_mvnx([file_dir, '/' file_name])
% Look at the comment to know which trial it is!

Subject_name = 'GuSe';
Move_name = ['4-o', '8-1o', '8--o'];

mvnx_converter_general_trampo(mvnx, file_dir, file_name, Subject_name, Move_name);

disp('Success')

















