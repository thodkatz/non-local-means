#! /bin/octave -qf

% FORKED BY https://github.com/AUTh-csal/pds-codebase/tree/main/matlab
%
% 1) APPLY GAUSSIAN NOISE TO AN IMAGE REPRESENTED AS A 2D ARRAY.
% 2) INVOKE THE OUTPUT OF A NON LOCAL MEANS FILTER IMPLEMENTED IN C.
% 3) RENDER THE 2D ARRAYS IN GRAYSCALE.

fprintf("\n--------------SCRIPT BEGINS--------------\n");
clear all
close all

pkg load image statistics

path = './data/';
strImgVar = 'house';
args = argv();

normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

% REQUIREMENT:
% THE INPUT OF AN IMAGE SHOULD BE REPRESENTED IN A 2D ARRAY (GRAYSCALE)
fprintf('Loading input data...\n')
ioImg = load([path strImgVar '.mat']);
I = ioImg.(strImgVar);
%I = dlmread([path 'house.txt']); 

% NORMALIZING
fprintf("Normalizing image (0 - 1)...\n");
I = normImg(I);

% APPLY NOISE
fprintf("Applying noise...\n");
J = imnoise(I, 'gaussian', 0, 0.001);
%J = dlmread([path 'noise_image_matlab.txt']);

fprintf("Input ready to be parsed from our C code\n");
dlmwrite([path 'noise_image.txt'], size(J), 'delimiter', ' ');
dlmwrite([path 'noise_image.txt'], J, '-append', 'delimiter', ' ', 'precision', '%.06f');

Flag = ' ';
if length(args) == 2
    Flag = args{2}
end

% NON LOCAL MEANS IMPLEMENTED IN C
fprintf("\n\033[1mC CODE LAUNCHED...\033[0m\n");
exe = ["./bin/" args{1}];
if length(args) == 2
    exe = ["./bin/" args{1} ' ' Flag];
end
system(exe);
If = dlmread([path 'filtered_image.txt']);
fprintf("\033[1mC CODE ENDED...\033[0m\n\n");

% NON LOCAL MEANS IMPLEMENTED IN OCTAVE/MATLAB
Params = dlmread([path 'parameters.txt']);
filtSigma = Params(1);
patchSize = [Params(2) Params(2)];
patchSigma = Params(3);

% VALIDATION
if Flag == '--debug'
    fprintf("Debugging...\n");
    cd src/octave/
    IfOctave = nonLocalMeans(J, patchSize, filtSigma, patchSigma);

    % CHECK IF PATCHES ARE THE SAME
    PatchesOctave = dlmread(['../../' path 'debug/patches_oct.txt']);
    PatchesOctave = ceil(100 .* PatchesOctave)./100;
    PatchesC      = dlmread(['../../' path 'debug/patches_c.txt']);
    PatchesC      = ceil(100 .* PatchesC)./100;

    if PatchesOctave == PatchesC
        fprintf("\x1B[32m \xE2\x9C\x94 Patches applied with gaussian weights\x1B[0m\n");
    elseif
        fprintf("\x1B[31m \xE2\x9D\x8C Patches applied with gaussian weights\x1B[0m\n"); 
        %dlmwrite(['../../' path 'debug/errors/.txt'], PatchesOctave, 'delimiter', ' ', 'precision', '%.02f');
        %dlmwrite(['../../' path 'debug/test2.txt'], PatchesC, 'delimiter', ' ', 'precision', '%.02f');
    end

    % CHECK DISTANCE MATRIX
    DistancesOctave = dlmread(['../../' path 'debug/distances_oct.txt']);
    DistancesOctave = ceil(10 .* DistancesOctave)./10;
    DistancesC      = dlmread(['../../' path 'debug/distances_c.txt']);
    DistancesC      = ceil(10 .* DistancesC)./10;

    if DistancesOctave == DistancesC
        fprintf("\x1B[32m \xE2\x9C\x94 Distances\x1B[0m\n"); 
    elseif
        fprintf("\x1B[31m \xE2\x9D\x8C Distances\x1B[0m\n");
        %dlmwrite(['../../' path 'debug/test1.txt'], DistancesOctave, 'delimiter', ' ', 'precision', '%.02f');
        %dlmwrite(['../../' path 'debug/test2.txt'], DistancesC, 'delimiter', ' ', 'precision', '%.02f');
    end

    % CHECK WEIGHTS
    WeightsOctave = dlmread(['../../' path 'debug/weights_oct.txt']);
    WeightsOctave = ceil(10 .* DistancesOctave)./10;
    WeightsC      = dlmread(['../../' path 'debug/weights_c.txt']);
    WeightsC      = ceil(10 .* DistancesC)./10;

    if WeightsOctave == WeightsC
        fprintf("\x1B[32m \xE2\x9C\x94 Weights\x1B[0m\n"); 
    elseif
        fprintf("\x1B[31m \xE2\x9D\x8C Weights\x1B[0m\n");
        %dlmwrite(['../../' path 'debug/test1.txt'], WeightsOctave, 'delimiter', ' ', 'precision', '%.02f');
        %dlmwrite(['../../' path 'debug/test2.txt'], WeightsC, 'delimiter', ' ', 'precision', '%.02f');
    end

    % CHECK FILTERED IMAGE
    FilteredOctave = dlmread(['../../' path 'debug/filtered_image_oct.txt']);
    FilteredOctave = ceil(100 .* FilteredOctave)./100;
    FilteredC      = dlmread(['../../' path 'debug/filtered_image_c.txt']);
    FilteredC      = ceil(100 .* FilteredC)./100;

    if FilteredOctave == FilteredC
        fprintf("\x1B[32m \xE2\x9C\x94 Filtering\x1B[0m\n"); 
    elseif
        fprintf("\x1B[31m \xE2\x9D\x8C Filtering\x1B[0m\n");
        %dlmwrite(['../../' path 'debug/test1.txt'], FilteredOctave, 'delimiter', ' ', 'precision', '%.02f');
        %dlmwrite(['../../' path 'debug/test2.txt'], FilteredC, 'delimiter', ' ', 'precision', '%.02f');
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
