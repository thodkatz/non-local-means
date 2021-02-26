#! /bin/octave -qf

% FORKED BY https://github.com/AUTh-csal/pds-codebase/tree/main/matlab
%
% 1) APPLY GAUSSIAN NOISE TO AN IMAGE REPRESENTED AS A 2D ARRAY.
% 2) INVOKE THE OUTPUT OF A NON LOCAL MEANS FILTER IMPLEMENTED IN C.
% 3) RENDER THE 2D ARRAYS IN GRAYSCALE.

fprintf("\n--------------SCRIPT BEGINS--------------\n");
fprintf("\033[1mFor this script to run successfully make sure to fulfill the requirements\033[0m\n");
clear all
close all

pkg load image statistics

path = './data/';
args = argv();

normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

fprintf('Loading input data...\n')

% WAYS TO READ AN IMAGE:

%strImgVar = 'lena512';
%ioImg = load([path strImgVar '.mat']);
%I = ioImg.(strImgVar);

%I = dlmread([path 'house.txt']); 

I = imread([path 'house.jpg']);
%I = rgb2gray(I);
%I = imresize(I, [128 128]);
%imwrite(I, [path 'tulips128.jpg']);
%I = reshape(I, [length(I) length(I)])

I = double(I);

% NORMALIZING
fprintf("Normalizing image (0 - 1)...\n");
I = normImg(I);

% APPLY NOISE
fprintf("Applying noise...\n");
J = imnoise(I, 'gaussian', 0, 0.002);
%J = dlmread([path 'noise_image_house_const.txt']);
%dlmwrite([path 'noise_image_house_const.txt'], J, 'delimiter', ' ', 'precision', '%.06f');

fprintf("Input ready to be parsed from our C code\n");
dlmwrite([path 'noise_image.txt'], size(J), 'delimiter', ' ');
dlmwrite([path 'noise_image.txt'], J, '-append', 'delimiter', ' ', 'precision', '%.06f');

DebugFlag = ' ';
if length(args) == 2
    DebugFlag = args{2}
end

% NON LOCAL MEANS IMPLEMENTED IN C
fprintf("\n\033[1mC CODE LAUNCHED...\033[0m\n");
exe = ["./bin/" args{1}];
if length(args) == 2
    exe = ["./bin/" args{1} ' ' DebugFlag];
end

% AS A NON NVIDIA USER I CANT USE THIS SCRIPT LOCALLY FOR CUDA VERSIONS
system(exe); 

% READ THE OUTPUT OF THE NON LOCAL MEANS IMPLEMENTED IN C
If = dlmread([path 'filtered_image' args{1} '.txt']);
fprintf("\033[1mC CODE ENDED...\033[0m\n\n");

% VALIDATION BASED ON v0
if DebugFlag == '--debug'
    fprintf("Validation...\n");

    %cd src/octave/
    % NON LOCAL MEANS IMPLEMENTED IN OCTAVE/MATLAB
    %IfOctave = nonLocalMeans(J, patchSize, filtSigma, patchSigma);

    % CHECK IF PATCHES ARE THE SAME
    Patches1      = dlmread([path 'debug/' args{1} '/patches_c.txt']);
    Patches2      = dlmread([path 'debug/v0/patches_c.txt']);
    ErrorPatches = abs(Patches1 - Patches2);
    ErrorPatches = max(ErrorPatches(:));

    if ErrorPatches < 0.001
        fprintf("\x1B[32m \xE2\x9C\x94 Patches \x1B[0m\n");
    elseif
        fprintf("\x1B[31m \xE2\x9D\x8C Patches \x1B[0m\n"); 
        fprintf("Patches error: %f\n", ErrorPatches);
    end

    % CHECK FILTERED IMAGE
    Filtered1      = dlmread([path 'debug/' args{1} '/filtered_image_c.txt']);
    Filtered2      = dlmread([path 'debug/v0/filtered_image_c.txt']);
    ErrorFiltering = abs(Filtered1 - Filtered2);
    ErrorFiltering = max(ErrorFiltering(:));

    if ErrorFiltering < 0.001
        fprintf("\x1B[32m \xE2\x9C\x94 Filtering \x1B[0m\n"); 
    elseif
        fprintf("\x1B[31m \xE2\x9D\x8C Filtering \x1B[0m\n");
        fprintf("Filtering error: %f\n", ErrorFiltering);
    end
end

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

fprintf("\n\033[1mPRESS ENTER TO CONTINUE...\033[0m");
pause() 

fprintf("--------------TO BE CONTINUED--------------\n");
