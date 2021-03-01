#! /bin/octave -qf

fprintf("\n--------------SCRIPT BEGINS--------------\n");
clear all
close all

pkg load image statistics

printf("\033[1m REQUIREMENT: Run first 'octave src/octave/jpg2array.m <image file>' and then the version ('./bin/<version> <arguments>') that its output you desire to render \033[0m \n");
printf("For example:\n");
printf("\033[1m octave src/octave/jpg2array data/house.jpg \033[0m \n");
printf("\033[1m ./bin/v0 data/noise_image_house_const.txt 3 \033[0m \n");
printf("\033[1m octave src/octave/rendering.m v0 \033[0m \n");

path = './data/';
args = argv();
version = args{1};

if length(args) < 1
    printf("\x1B[31mBad arguments\x1B[0m\n");
    exit;
end


% REQUIREMENTS
I = dlmread([path 'clear_image.txt']);
J = dlmread([path 'noise_image.txt'], ' ', 1, 0);

% CUDA PROGRAM HAS BEEN ALREADY EXECUTED 
If = dlmread([path 'filtered_image_' version '.txt']);

% RENDERING
figure('Name','Original Image');
imagesc(I); axis image;
colormap gray;

figure('Name','Noisy-Input Image');
imagesc(J); axis image;
colormap gray;

figure('Name', 'Filtered image');
imagesc(If); axis image;
colormap gray;

figure('Name', 'Residual');
imagesc(If-J); axis image;
colormap gray;

fprintf("\n\033[1mPRESS ENTER TO CONTINUE...\033[0m\n");
pause() 

fprintf("--------------TO BE CONTINUED--------------\n");
