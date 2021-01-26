#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "v0.h"
#include "utils.h"

/*
 * For the pixels in the edges, patch  size extends image.
 * Fill the missing values with this flag/macro.
 *
 * TODO: how to make use of missing values in "special" patches? Wrap the image?
 */
#define OUT_OF_BOUNDS -1.0

// expecting only even number of patch size
float *non_local_means(int m, int n, float *noise_image, int patch_size, float filt_sigma, float patch_sigma, int argc, char *argv[]) {
    struct timespec tic;
    struct timespec toc;
    int total_patch_size = patch_size * patch_size;
    int total_pixels = m * n;

    printf("\nCreating patches...\n");
    TIC()
    float *patches;
    patches = create_patches(noise_image, patch_size, m, n);
    TOC("Time elapsed creating patches: %lf\n")

    float *gauss_patch;
    gauss_patch = create_gauss_kernel(patch_size, patch_sigma);

    // apply gaussian patch
   
    printf("\nApplying gaussian patch...\n");
    TIC()
    for(int i = 0; i < total_pixels; i ++) {
        for(int k = 0; k < total_patch_size; k++) {
            if(patches[i*total_patch_size + k] != OUT_OF_BOUNDS) {
                patches[i*total_patch_size + k] *= gauss_patch[k];
            }
        }
    }
    TOC("Time elapsed applying guassian patch: %lf\n")

    float *filtered_image;
    MALLOC(float, filtered_image, total_pixels);
    printf("Filtering...\n");
    filtering(patches, patch_size, filt_sigma, noise_image, total_pixels, filtered_image);

    // debugging
    
    FILE *debug_patches;
    if(argc == 2 && strcmp(argv[1],"--debug") == 0) {
        printf("Writing patches to file. Mode: \033[1mdebug\033[0m...\n");
        debug_patches = fopen("data/debug/v0/patches_c.txt", "w");
        print_patch_file(debug_patches, patches, patch_size, m*n);
        fclose(debug_patches);
    }

    FILE *debug_filtering;
    if(argc == 2 && strcmp(argv[1],"--debug") == 0) {
        printf("Writing filtering image to file. Mode: \033[1mdebug\033[0m...\n");
        debug_filtering = fopen("data/debug/v0/filtered_image_c.txt", "w");
        print_array_file(debug_filtering, filtered_image, m, n);
        fclose(debug_filtering);
    }

    free(patches);
    free(gauss_patch);

    return filtered_image;
}


void filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, int total_pixels, float *filtered_image) {
    for(int pixel = 0; pixel < total_pixels; pixel++) {
        printf("Pixelth: %d of %d\n", pixel, total_pixels);

        float *weights;
        MALLOC(float, weights, total_pixels);
        weights = euclidean_distance_matrix_per_pixel(patches, patch_size, pixel, total_pixels);

        float max = -1.0;
        float sum_weights = 0;
        for(int k = 0; k<total_pixels; k++) {
            weights[k] = exp(-pow(weights[k], 2) / filt_sigma);
            if(weights[k] > max && pixel!=k) max = weights[k];
            if(pixel!=k) sum_weights += weights[k];
        }

        weights[pixel] = max;
        sum_weights += max;

        filtered_image[pixel] = apply_weighted_pixels(weights, noise_image, total_pixels);
        filtered_image[pixel] /= sum_weights;

        free(weights);
    }
}
