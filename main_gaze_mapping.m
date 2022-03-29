% this code allows to create a map of the gaze trajectory from Xsens +
% eye-tracker measurements on trampolinists


clear all; close all; clc;

file_dir = '/home/user/Documents/Programmation/gaze_trajectory_Pupil_Xsens/XsensData';
file_name = 'Test_17032021-003.mvnx';

mvnx = load_mvnx([file_dir, '/' file_name])
% Look at the comment to know which trial it is!

Subject_name = 'GuSe';
Move_name = ['4-/', '43/', '4-/', '43/', '4-/', '43/', '4-/', '43/', '4-/', '43/'];

mvnx_converter_general_trampo(mvnx, file_dir, file_name);

disp('coucou')

















