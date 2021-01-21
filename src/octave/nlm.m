#! /bin/octave -qf

% Forked by 
%
% 1) APPLY GAUSSIAN NOISE TO AN IMAGE REPRESENTED AS A 2D ARRAY.
% 2) INVOKE THE OUTPUT OF A NON LOCAL MEANS FILTER IMPLEMENTED IN C.
% 3) RENDER THE 2D ARRAYS IN GRAYSCALE.

fprintf("--------------SCRIPT BEGINS--------------\n");
pkg load image statistics

path = './data/';
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

fprintf('...begin %s...\n',mfilename);  

% REQUIREMENT:
% THE INPUT OF AN IMAGE SHOULD BE REPRESENTED IN A 2D ARRAY (GRAYSCALE)
fprintf('...loading input data...\n')
I = dlmread(strcat(path, 'house.txt')); 

fprintf("Input ready to be parsed from our C code\n");
dlmwrite(strcat(path, 'house_c.txt'), size(I), 'delimiter', ' ');
dlmwrite(strcat(path, 'house_c.txt'), I, '-append', 'delimiter', ' ');

% NORMALIZING
fprintf('Normalizing image... 0 - 1 \n')
I = normImg(I);

% APPLY NOISE
J = imnoise(I, 'gaussian', 0, 0.001);

% NON LOCAL MEANS OUTPUT IMPLEMENTED IN C
fprintf("C code launched...\n")
args = argv();
exe = strcat("./bin/", args{1});
system(exe);
If = dlmread(strcat(path, 'nlm_test.txt'));

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

fprintf('...end %s...\n',mfilename);

fprintf("PRESS ENTER TO CONTINUE...")
pause() 

fprintf("--------------TO BE CONTINUED--------------\n");
