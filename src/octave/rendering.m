#! /bin/octave -qf

fprintf("\n--------------SCRIPT BEGINS--------------\n");
clear all
close all

pkg load image statistics

printf("\033[1mREQUIREMENT: Run first jpg2array.m to create necessarry input\033[0m\n");
printf("USAGE: octave src/octave/rendering.m <version>\n");
printf("e.g. octave src/octave/rendering.m v1\n");
printf("+ optional flag: --debug\n");
printf("e.g. octave src/octave/rendering.m v1 --debug\n");

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

% IT IS SUPPOSED TO BE PROVIDED A .txt FILE CREATED BY A THRID PARTY CUDA CAPABLE MACHINE FOR v1 and v2
If = dlmread([path 'filtered_image_' version '.txt']);

DebugFlag = ' ';
if length(args) == 2
    DebugFlag = args{2};
end

% VALIDATION
if DebugFlag == '--debug'
    fprintf("Validation...\n");

    % CHECK IF PATCHES ARE THE SAME
    Patches1      = dlmread([path 'debug/' version '/patches_c.txt']);
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
    Filtered1      = dlmread([path 'debug/' version '/filtered_image_c.txt']);
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

fprintf("\n\033[1mPRESS ENTER TO CONTINUE...\033[0m\n");
pause() 

fprintf("--------------TO BE CONTINUED--------------\n");
