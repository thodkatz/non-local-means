#include <stdio.h>
#include <math.h>
#include <time.h>
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
float *non_local_means(int m, int n, float *noise_image, int patch_size, float filt_sigma, float patch_sigma) {
    struct timespec tic;
    struct timespec toc;

    float *filtered_image;
    MALLOC(float, filtered_image, m * n);

    int total_patch_size = patch_size * patch_size;

    float *patches;
    patches = create_patches(noise_image, patch_size, m, n);

    print_patch(patches, patch_size, m * n);

    float *gauss_patch;
    gauss_patch = create_gauss_kernel(patch_size, patch_sigma);

    printf("\nGaussian patch...\n");
    print_patch(gauss_patch, patch_size, 1);

    // apply gaussian patch
   
    for(int i = 0; i < m * n; i ++) {
        for(int k = 0; k < total_patch_size; k++) {
            if(patches[i*total_patch_size + k] != OUT_OF_BOUNDS) {
                patches[i*total_patch_size + k] *= gauss_patch[k];
            }
        }
    }

    printf("\nPatches after gaussian weights\n");
    print_patch(patches, patch_size, m*n);

    // calculate distances
    
    float *sum_weights;
    CALLOC(float, sum_weights, m*n);
    float *weights;
    printf("Calculating distance matrix...\n");
    TIC();
    weights = euclidean_distance_matrix(patches, patch_size, m, n);
    TOC("Time elapsed calculating distance matrix: %lf\n")

    printf("Distances between patches per patch\n");
    print_array(weights , m*n, m*n);

    // weight formula per patch: D = exp(-D.^2 / filt_sigma)
    
    printf("Calculating weights...\n");
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
    TOC("Time elapsed calculating weights: %lf\n")

    print_array(weights, m*n, m*n);

    // filtering
    
    printf("Filtering...\n");
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            filtered_image[i*n + j] = apply_weighted_pixels(weights + i*(m*n)*n + j*(m*n), noise_image, m*n);
            filtered_image[i*n + j] /= sum_weights[i*n + j];
        }
    }

    printf("Filtered image\n");
    print_array(filtered_image, m, n);


    free(patches);
    free(gauss_patch);
    free(weights);
    free(sum_weights);

    return filtered_image;
}

float *create_patches(float *image, int patch_size, int m, int n) {
    int total_patch_size = patch_size * patch_size;

    float *patches;
    MALLOC(float, patches, m*n*total_patch_size);
    for(int i = 0; i < m*n*total_patch_size; i++) patches[i] = OUT_OF_BOUNDS;

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int top_left_corner_x = i - (patch_size - 1)/2;
            int top_left_corner_y = j - (patch_size - 1)/2;
            int patch_num = i*n + j;
            for(int patch_x = top_left_corner_x, patch_id = 0; patch_x < top_left_corner_x + patch_size; patch_x++) {
                for(int patch_y = top_left_corner_y; patch_y < top_left_corner_y + patch_size; patch_y++) {
                    if((0<= patch_x && patch_x <= m-1) && (0<= patch_y && patch_y <= n-1)) {
                        int pixel = patch_x*n + patch_y;
                        patches[patch_num*total_patch_size + patch_id] = image[pixel];
                    }
                    patch_id++;
                }
            }
        }
    }

    return patches;
}

float *create_gauss_kernel(int patch_size, float patch_sigma) {
    int total_patch_size = patch_size * patch_size;

    float *gauss_patch;
    MALLOC(float, gauss_patch, total_patch_size);
    //float gauss_sum = 0;
    
    for(int i = 0; i < patch_size; i++) {
        for(int j = 0; j < patch_size; j ++) {
            int x = i - (patch_size-1)/2;
            int y = j - (patch_size-1)/2;
            gauss_patch[i*patch_size + j] = exp(-(x * x + y * y) / (2 * patch_sigma * patch_sigma));
            //gauss_sum += gauss_patch[i*patch_size + j];
        }
    }

    return gauss_patch;
}

// nearness is determined by how similar is the intensity of the pixels
float *euclidean_distance_matrix(float *patches, int patch_size, int m, int n) {
    int total_patch_size = patch_size * patch_size;
    int len_mat = m*n; // symmetric (m*n x m*n)

    float *distance;
    MALLOC(float, distance, m*n * m*n);

    for(int i = 0; i < len_mat; i++) {
        for(int j = 0; j < len_mat; j++) {
            distance[i*(len_mat) + j] = euclidean_distance_patch(patches + i*total_patch_size, patches + j*total_patch_size, patch_size);
        }
    }

    return distance;
}

// take two patches and calculate their distance
float euclidean_distance_patch(float *patch1, float *patch2, int patch_size) {
    int total_patch_size = patch_size * patch_size;

    float distance = 0;
    for(int i = 0; i < total_patch_size; i++) {
        if(patch1[i] != OUT_OF_BOUNDS && patch2[i] != OUT_OF_BOUNDS) {
            distance += pow(patch1[i] - patch2[i], 2); 
        }
    }

    return distance;
}

float apply_weighted_pixels(float *weights, float *image, int image_size) {
    float new_pixel = 0;

    for(int i = 0; i < image_size; i++) {
        new_pixel += weights[i] * image[i];
    }

    return new_pixel;
}
