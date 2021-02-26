#! /bin/octave -qf

fprintf("\n--------------CREATE A NOISED IMAGE--------------\n");
clear all
close all

pkg load image statistics

path = './data/';
args = argv();

printf("USAGE: octave src/octave/jpg2array.m data/<name of image>.jpg\n");
printf("e.g. octave src/octave/jpg2array.m data/house.jpg\n");

if length(args) < 1
    printf("\x1B[31mBad arguments\x1B[0m\n");
    exit;
end

normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

fprintf('Loading input data...\n')

% DEPENDING THE TYPE OF THE IMAGE YOU NEED TO TUNE IT A LITTLE BIT. RGB? SIZE?

image = args{1};
I = imread(image);
%I = rgb2gray(I);
%I = imresize(I, [128 128]);
%imwrite(I, [path 'tulips128.jpg']);
I = double(I);
%I = reshape(I, [length(I) length(I)])

% NORMALIZING
fprintf("Normalizing image (0 - 1)...\n");
I = normImg(I);

dlmwrite([path 'clear_image.txt'], I, 'delimiter', ' ');
fprintf("Created data/clear_image.txt\n");

% APPLY NOISE
fprintf("Applying noise...\n");
J = imnoise(I, 'gaussian', 0, 0.002);

dlmwrite([path 'noise_image.txt'], size(J), 'delimiter', ' ');
dlmwrite([path 'noise_image.txt'], J, '-append', 'delimiter', ' ', 'precision', '%.06f');
fprintf("Created data/noise_image.txt\n");
