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

if length(args) < 3
    printf("Bad arguments\n");
    exit;
end

version = args{1};
imageFile = args{2};
patchSize = args{3};

if (version == 'v1' || version == 'v2') && length(args) < 5
    printf("Bad arguments\n");
    exit;
end;

# For v1,v2 validation check
DebugFlag = ' ';
if length(args) == 6
    DebugFlag = args{6};
end

normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

fprintf('Loading input data...\n')

% WAYS TO READ AN IMAGE:

%strImgVar = 'lena512';
%ioImg = load([path strImgVar '.mat']);
%I = ioImg.(strImgVar);

%I = dlmread([path 'house.txt']); 

I = imread(imageFile);
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

dlmwrite([path 'noise_image.txt'], size(J), 'delimiter', ' ');
dlmwrite([path 'noise_image.txt'], J, '-append', 'delimiter', ' ', 'precision', '%.06f');

fprintf("\n\033[1mC CODE LAUNCHED...\033[0m\n");
if version == 'v0'
    exe = ["./bin/" version ' ' path "noise_image.txt " patchSize];
else
    exe = ["./bin/" version ' ' path "noise_image.txt " patchSize ' ' args{4} ' ' args{5}];
    if length(args) == 6 && args{6} == "--debug"
        exe = [exe " --debug"];
    end
end
system(["make " version]);
system(exe); 

% READ THE OUTPUT OF THE NON LOCAL MEANS IMPLEMENTED IN C
If = dlmread([path 'filtered_image_' version '.txt']);
fprintf("\033[1mC CODE ENDED...\033[0m\n\n");

% VALIDATION BASED ON v0
if DebugFlag == '--debug'
    fprintf("Validation...\n");

    system(["./bin/v0" path "noise_image.txt " patchSize ' ' DebugFlag]);

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

fprintf("\n\033[1mPRESS ENTER TO CONTINUE...\033[0m");
pause() 

fprintf("--------------TO BE CONTINUED--------------\n");