#include <stdio.h>
#include "v0.h"
#include "utils.h"

/*
 * For the pixels in the edges, patch  size extends image.
 * Fill the missing values with this flag/macro.
 *
 * TODO: how to make use of missing values in "special" patches? Wrap the image?
 */
#define OUT_OF_BOUNDS -1.0

float *non_local_means(int m, int n, float *noise_image, int patch_size, float filt_sigma, float patch_sigma) {
    float *filtered_image;
    MALLOC(float, filtered_image, m * n);
    for(int i = 0; i < m * n; i++) filtered_image[i] = 369;

    int total_patch_size = patch_size * patch_size;

    float *patches;
    MALLOC(float, patches, m*n*total_patch_size);
    for(int i = 0; i < m*n*total_patch_size; i++) patches[i] = OUT_OF_BOUNDS;

    // create patches
    int patch_num = 0;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int top_left_corner_x = i - (patch_size - 1)/2;
            int top_left_corner_y = j - (patch_size - 1)/2;

            for(int patch_x = 0; patch_x < patch_size; patch_x++) {
                int pixel = top_left_corner_x*n + top_left_corner_y;
                for(int patch_y = 0; patch_y < patch_size; patch_y++) {
                    int patch_id = patch_num*(patch_x * patch_size + patch_y);
                    pixel += patch_y;
                    if(pixel >= 0 && pixel <= (m * n - 1)) {
                        patches[patch_id] = noise_image[pixel];
                    }
                }
                top_left_corner_x++;
                top_left_corner_y = j - (patch_size - 1)/2;
            }
            patch_num++;
        }
    }

    // create gaussian patch patch_size x patch_size
    // row major
    float *gaussian_patch;
    MALLOC(float, gaussian_patch, total_patch_size);
    // row wise gaussian distribution values

    // apply gaussian patch
    patch_num = 0;
    for(int i = 0; i < m; i ++) {
        for(int j = 0; j < n; j++) {
            int patch_id = patch_num * (i*n + j);
            for(int k = 0; k < total_patch_size; k++) {
                patches[patch_id + k] *= gaussian_patch[k];
            }
            patch_num++;
        }
    }

    // euclidean distance between patches

    // weigth formula per patch
    // D = exp(-D.^2 / filt_sigma)

    // filtering
    // I could probably use cblas? But we will use CUDA for this...


    free(patches);
    free(gaussian_patch);
    return filtered_image;
}
