% 1) APPLY GAUSSIAN NOISE TO AN IMAGE REPRESENTED AS A 2D ARRAY.
% 2) INVOKE THE OUTPUT OF A NON LOCAL MEANS FILTER IMPLEMENTED IN C TO FIX THE IMAGE.
% 3) RENDER THE 2D ARRAYS IN GRAYSCALE.

pkg load image

path = './data/';
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

fprintf('...begin %s...\n',mfilename);  

% CAN OCTAVE WORK WITH .MAT FILES?
% REQUIREMENT:
% THE INPUT OF AN IMAGE SHOULD BE REPRESENTED IN A 2D ARRAY. LATER IT WILL BE NORMALIZED TO OUR NEEDS.
fprintf('...loading input data...\n')
I = dlmread(strcat(path, 'house.txt')); 

fprintf("Now input is ready to be parsed from our C code\n");
dlmwrite(strcat(path, 'house_c.txt'), size(I), 'delimiter', ' ');
dlmwrite(strcat(path, 'house_c.txt'), I, '-append', 'delimiter', ' ');

fprintf(' - normalizing image...\n')
I = normImg(I);

% APPLY NOISE
J = imnoise(I, 'gaussian', 0, 0.001);

% NON LOCAL MEANS OUTPUT IMPLEMENTED IN C
fprintf("C code launched...\n")
%system(./bin/v0.c)
If = dlmread(strcat(path, 'nlm.txt'));

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

% for some reason the last image wont popup 
figure('Name', 'Residual');
imagesc(If-J); axis image;
colormap gray;

fprintf('...end %s...\n',mfilename);
