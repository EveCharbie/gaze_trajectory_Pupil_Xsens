% this code allows to create a map of the gaze trajectory from Xsens +
% eye-tracker measurements on trampolinists


clear all; close all; clc;

file_dir = '/home/user/Documents/Eye-tracking/CollecteVision/XsensData/exports/kin_jenn';
file_name = 'Test_17032021-007.mvnx';

mvnx = load_mvnx([file_dir, '/' file_name])
% Look at the comment to know which trial it is!

Subject_name = 'GuSe';
Move_name = ['4-o', '8-1<', '811<'];

data = mvnx_converter_general_trampo(mvnx);

disp('coucou')

















