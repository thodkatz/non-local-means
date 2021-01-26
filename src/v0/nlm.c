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

    printf("\nCreating patches...\n");
    TIC();
    float *patches;
    patches = create_patches(noise_image, patch_size, m, n);
    TOC("Time elapsed creating patches: %lf\n");

    float *gauss_patch;
    gauss_patch = create_gauss_kernel(patch_size, patch_sigma);

    // apply gaussian patch
   
    printf("\nApplying gaussian patch...\n");
    TIC();
    for(int i = 0; i < m * n; i ++) {
        for(int k = 0; k < total_patch_size; k++) {
            if(patches[i*total_patch_size + k] != OUT_OF_BOUNDS) {
                patches[i*total_patch_size + k] *= gauss_patch[k];
            }
        }
    }
    TOC("Time elapsed applying guassian patch: %lf\n");


    // calculate distances
    
    float *sum_weights;
    CALLOC(float, sum_weights, m*n);
    float *weights;
    printf("\nCalculating distance matrix symmetric...\n");
    TIC();
    weights = euclidean_distance_symmetric_matrix(patches, patch_size, m*n, m*n);
    TOC("Time elapsed calculating distance matrix: %lf\n");

    // weight formula per patch: D = exp(-D.^2 / filt_sigma)
    
    printf("\nCalculating weights...\n");
    TIC();
    for(int i = 0; i < m*n; i++){
        float max  = -1.0; 
        for(int j = 0; j < m*n; j++) {
            weights[i*(m*n) + j] = exp(-pow(weights[i*(m*n) + j], 2) / filt_sigma);
            if(weights[i*(m*n) + j] > max && j!=i) max = weights[i*(m*n) + j];
            if(j!=i) sum_weights[i] += weights[i*(m*n) + j];
        }
        // normalize the self distance (weigth 1.0) with the maximum value of the compared patches per patch
        weights[i*(m*n) + i] = max; 
        sum_weights[i] += max;
    }
    TOC("Time elapsed calculating weights: %lf\n");

    // filtering
    
    float *filtered_image;
    MALLOC(float, filtered_image, m * n);

    TIC();
    printf("\nFiltering...\n");
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            filtered_image[i*n + j] = apply_weighted_pixels(weights + i*(m*n)*n + j*(m*n), noise_image, m*n);
            filtered_image[i*n + j] /= sum_weights[i*n + j];
        }
    }
    TOC("Time elapsed filtering image: %lf\n");

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
    free(weights);
    free(sum_weights);

    return filtered_image;
}

float *euclidean_distance_symmetric_matrix(float *patches, int patch_size, int rows, int cols) {
    int total_patch_size = patch_size * patch_size;

    float *distance;
    MALLOC(float, distance, rows * cols);

    for(int i = 0; i < rows; i++) {
        for(int j = i; j < cols; j++) {
            distance[i*cols + j] = euclidean_distance_patch(patches + i*total_patch_size, patches + j*total_patch_size, patch_size);
            distance[j*cols + i] = distance[i*cols + j];
        }
    }

    return distance;
}
