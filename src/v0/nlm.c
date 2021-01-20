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
    for(int i = 0; i < m * n; i++) filtered_image[i] = 369; // dummy init

    int total_patch_size = patch_size * patch_size;

    float *patches;
    MALLOC(float, patches, m*n*total_patch_size);
    for(int i = 0; i < m*n*total_patch_size; i++) patches[i] = OUT_OF_BOUNDS;

    // create patches
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int top_left_corner_x = i - (patch_size - 1)/2;
            int top_left_corner_y = j - (patch_size - 1)/2;
            int patch_num = i*n + j;
            for(int patch_x = top_left_corner_x, patch_id = 0; patch_x < top_left_corner_x + patch_size; patch_x++) {
                for(int patch_y = top_left_corner_y; patch_y < top_left_corner_y + patch_size; patch_y++) {
                    if((0<= patch_x && patch_x <= m) && (0<= patch_y && patch_y <= n)) {
                        int pixel = patch_x*n + patch_y;
                        patches[patch_num*total_patch_size + patch_id] = noise_image[pixel];
                    }
                    patch_id++;
                }
            }
        }
    }

    print_patch(patches, patch_size, m * n);

    // create gaussian patch patch_size x patch_size
    // row major
    float *gaussian_patch;
    MALLOC(float, gaussian_patch, total_patch_size);
    // row wise gaussian distribution values
    // it is called gaussian kernel

    // apply gaussian patch
    for(int i = 0; i < m * n; i ++) {
        for(int k = 0; k < total_patch_size; k++) {
            patches[i*total_patch_size + k] *= gaussian_patch[k];
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
