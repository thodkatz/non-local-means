function If = nonLocalMeans(I, patchSize, filtSigma, patchSigma)

    % FORKED BY https://github.com/AUTh-csal/pds-codebase/tree/main/matlab
    %
    % NONLOCALMEANS - Non local means CPU implementation
    %   
    % SYNTAX
    %
    %   IF = NONLOCALMEANS( IN, FILTSIGMA, PATCHSIGMA )
    %
    % INPUT
    %
    %   IN          Input image                     [m-by-n]
    %   PATCHSIZE   Neighborhood size in pixels     [1-by-2]
    %   FILTSIGMA   Filter sigma value              [scalar]
    %   PATCHSIGMA  Patch sigma value               [scalar]
    %
    % OUTPUT
    %
    %   IF          Filtered image after nlm        [m-by-n]
    %
    % DESCRIPTION
    %
    %   IF = NONLOCALMEANS( IN, PATCHSIZE, FILTSIGMA, PATCHSIGMA ) applies
    %   non local means algorithm with sigma value of FILTSIGMA, using a
    %   Gaussian patch of size PATCHSIZE with sigma value of PATCHSIGMA.
    %
    %

    %% USEFUL FUNCTIONS

    % create 3-D cube with local patches
    patchCube = @(X,w) ...
        permute( ...
        reshape( ...
        im2col( ...
        padarray( ...
        X, ...
        (w-1)./2, 'symmetric'), ...
        w, 'sliding' ), ...
        [prod(w) size(X)] ), ...
        [2 3 1] );

    pathLogs = '../../data/debug/';

    %P = padarray(I, (patchSize-1)/2, 'symmetric');
    %dlmwrite("assert/padded_image.txt", P, 'delimiter', ' ', 'precision', '%.05f');

    % create 3D cube
    B = patchCube(I, patchSize);
    [m, n, d] = size( B );
    B = reshape(B, [ m*n d ] );
    %dlmwrite("assert/b.txt", B);

    for i = 1:m
        for j = 1:n
            patch = B((j-1)*m + i,:);
            patch = reshape(patch, patchSize);
            if i == 1 && j == 1
                %dlmwrite("assert/patches.txt", patch, 'delimiter', ' ', 'precision', '%.05f');
            else
                %dlmwrite("assert/patches.txt", patch, '-append','delimiter', ' ', 'precision', '%.05f');
            end
            if (i+j ~= m+n)
                %dlmwrite("assert/patches.txt", '', '-append','delimiter', ' ', 'precision', '%.05f');
            end
        end
    end
    %dlmwrite("assert/patches.txt", B, 'delimiter', ' ', 'precision', '%.05f');

    %dummy = dlmread("assert/patches.txt");

    % gaussian patch
    H = fspecial('gaussian',patchSize, patchSigma);
    H = H(:) ./ max(H(:));

    gaussKernel = reshape(H, patchSize);
    %dlmwrite("assert/gauss_kernel.txt", gaussKernel, 'delimiter', ' ', 'precision', '%.05f');

    % apply gaussian patch on 3D cube
    B = bsxfun( @times, B, H' );

    for i = 1:m
        for j = 1:n
            patch = B((j-1)*m + i,:);
            if i == 1 && j == 1
                B_row = patch;
            else
                B_row = [B_row ; patch];
            end
            patch = reshape(patch, patchSize);
            if i == 1 && j == 1
                dlmwrite([pathLogs "patches_oct.txt"], patch, 'delimiter', ' ', 'precision', '%.05f');
            else
                dlmwrite([pathLogs "patches_oct.txt"], patch, '-append','delimiter', ' ', 'precision', '%.05f');
            end
            if (i+j ~= m+n)
                dlmwrite([pathLogs "patches_oct.txt"], '', '-append','delimiter', ' ', 'precision', '%.05f');
            end
        end
    end

    % compute kernel
    D = squareform( pdist( B, 'euclidean' ) );
    D_row = squareform( pdist( B_row, 'euclidean' ) );
    dlmwrite([pathLogs "distances_oct.txt"], D_row, 'delimiter', ' ', 'precision', '%.05f');
    D = exp( -D.^2 / filtSigma );
    D_row = exp( -D_row.^2 / filtSigma );
    D(1:length(D)+1:end) = max(max(D-diag(diag(D)),[],2), eps);
    D_row(1:length(D_row)+1:end) = max(max(D_row-diag(diag(D_row)),[],2), eps);
    dlmwrite([pathLogs "weights_oct.txt"], D_row, 'delimiter', ' ', 'precision', '%.05f');

    % generate filtered image
    If = D*I(:) ./ sum(D, 2);

    % reshape for image
    If = reshape( If, [m n] );
    dlmwrite([pathLogs "filtered_image_oct.txt"], If, 'delimiter', ' ', 'precision', '%.05f');

end


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.2 - January 05, 2017
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%   0.2 (Jan 05, 2017) - Dimitris
%       * minor fix (distance squared)
%
% ------------------------------------------------------------

