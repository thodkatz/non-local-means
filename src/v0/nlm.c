#include <stdio.h>
#include "v0.h"
#include "utils.h"
#include <math.h>

/*
 * For the pixels in the edges, patch  size extends image.
 * Fill the missing values with this flag/macro.
 *
 * TODO: how to make use of missing values in "special" patches? Wrap the image?
 */
#define OUT_OF_BOUNDS -1.0

// expecting only even number of patch size
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

    /*
     * Gaussian kernel symmetric
     *
     * source:
     * - https://stackoverflow.com/questions/1696113/how-do-i-gaussian-blur-an-image-without-using-any-in-built-gaussian-functions
     * - https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
     * - https://stackoverflow.com/questions/54614167/trying-to-implement-gaussian-filter-in-c
     */
    float *gauss_patch;
    MALLOC(float, gauss_patch, total_patch_size);
    float gauss_sum = 0;
    for(int i = 0; i < patch_size; i++) {
        for(int j = 0; j < patch_size; j ++) {
            int x = i - (patch_size-1)/2;
            int y = j - (patch_size-1)/2;
            gauss_patch[i*patch_size + j] = exp(-(x * x + y * y) / (2 * patch_sigma * patch_sigma));
            gauss_sum += gauss_patch[i*patch_size + j];
        }
    }

    printf("\nGaussian patch no normalization\n");
    print_patch(gauss_patch, patch_size, 1);

    printf("sum %f\n", gauss_sum);

    float gauss_max = -1.0;
    for(int i = 0; i < total_patch_size; i++) {
        gauss_patch[i] /= gauss_sum; 
        if(gauss_max < gauss_patch[i]) {
            gauss_max = gauss_patch[i];
        }
    }

    printf("max %f\n", gauss_max);

    printf("\nGaussian patch...\n");
    print_patch(gauss_patch, patch_size, 1);

    // do we need to normalize with max?
    for(int i = 0; i < total_patch_size; i++) gauss_patch[i] /= gauss_max;

    printf("\nGaussian patch with normalization (max)\n");
    print_patch(gauss_patch, patch_size, 1);

    // apply gaussian patch
    for(int i = 0; i < m * n; i ++) {
        for(int k = 0; k < total_patch_size; k++) {
            patches[i*total_patch_size + k] *= gauss_patch[k];
        }
    }

    // euclidean distance between patches

    // weigth formula per patch
    // D = exp(-D.^2 / filt_sigma)

    // filtering


    free(patches);
    free(gauss_patch);

    return filtered_image;
}
