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
    TIC()
    float *patches;
    patches = create_patches(noise_image, patch_size, m, n);
    TOC("Time elapsed creating patches: %lf\n")
    //printf("Print patches...\n");
    //print_patch(patches, patch_size, m * n);

    float *gauss_patch;
    gauss_patch = create_gauss_kernel(patch_size, patch_sigma);
    //printf("\nGaussian patch...\n");
    //print_patch(gauss_patch, patch_size, 1);

    // apply gaussian patch
   
    printf("\nApplying gaussian patch...\n");
    TIC()
    for(int i = 0; i < m * n; i ++) {
        for(int k = 0; k < total_patch_size; k++) {
            if(patches[i*total_patch_size + k] != OUT_OF_BOUNDS) {
                patches[i*total_patch_size + k] *= gauss_patch[k];
            }
        }
    }
    TOC("Time elapsed applying guassian patch: %lf\n")

    //printf("\nPatches after gaussian weights\n");
    //print_patch(patches, patch_size, m*n);

    FILE *debug_patches;
    if(argc == 2 && strcmp(argv[1],"--debug") == 0) {
        printf("Writing patches to file. Mode: \033[1mdebug\033[0m...\n");
        debug_patches = fopen("data/debug/patches_c.txt", "w");
        print_patch_file(debug_patches, patches, patch_size, m*n);
        fclose(debug_patches);
    }

    // calculate distances
    
    float *sum_weights;
    CALLOC(float, sum_weights, m*n);
    float *weights;
    printf("\nCalculating distance matrix...\n");
    TIC()
    weights = euclidean_distance_matrix(patches, patch_size, m, n);
    TOC("Time elapsed calculating distance matrix: %lf\n")

    //printf("Distances between patches per patch\n");
    //print_array(weights , m*n, m*n);

    FILE *debug_distances;
    if(argc == 2 && strcmp(argv[1],"--debug") == 0) {
        printf("Writing distances to file. Mode: \033[1mdebug\033[0m...\n");
        debug_distances = fopen("data/debug/distances_c.txt", "w");
        print_array_file(debug_patches, weights, m*n, m*n);
        fclose(debug_distances);
    }

    // weight formula per patch: D = exp(-D.^2 / filt_sigma)
    
    printf("\nCalculating weights...\n");
    TIC()
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

    //print_array(weights, m*n, m*n);
    FILE *debug_weights;
    if(argc == 2 && strcmp(argv[1],"--debug") == 0) {
        printf("Writing weights to file. Mode: \033[1mdebug\033[0m...\n");
        debug_weights = fopen("data/debug/weights_c.txt", "w");
        print_array_file(debug_weights, weights, m*n, m*n);
        fclose(debug_weights);
    }
    // filtering
    
    float *filtered_image;
    MALLOC(float, filtered_image, m * n);

    TIC()
    printf("\nFiltering...\n");
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            filtered_image[i*n + j] = apply_weighted_pixels(weights + i*(m*n)*n + j*(m*n), noise_image, m*n);
            filtered_image[i*n + j] /= sum_weights[i*n + j];
        }
    }
    TOC("Time elapsed filtering image: %lf\n")

    //printf("Filtered image\n");
    //print_array(filtered_image, m, n);
    FILE *debug_filtering;
    if(argc == 2 && strcmp(argv[1],"--debug") == 0) {
        printf("Writing filtering image to file. Mode: \033[1mdebug\033[0m...\n");
        debug_filtering = fopen("data/debug/filtered_image_c.txt", "w");
        print_array_file(debug_filtering, filtered_image, m, n);
        fclose(debug_filtering);
    }

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

    // padding noise image symmetric
    float *image_padded;
    image_padded = padding_image(image, m, n, patch_size);

    /* printf("Padding image...\n"); */
    /* print_array(image_padded, m + (patch_size-1), n + (patch_size-1)); */
    /* printf("\n"); */

    int col_pad = (patch_size-1)/2;

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int top_left_corner_x = i;
            int top_left_corner_y = j;
            int patch_num = i*n + j;
            for(int patch_x = top_left_corner_x, patch_id = 0; patch_x < top_left_corner_x + patch_size; patch_x++) { 
                for(int patch_y = top_left_corner_y; patch_y < top_left_corner_y + patch_size; patch_y++) {
                    int pixel = patch_x*(n + 2*col_pad) + patch_y;
                    patches[patch_num*total_patch_size + patch_id] = image_padded[pixel];
                    patch_id++;
                }
            }
        }
    }

    free(image_padded);

    return patches;
}

/*
 * source: https://www.programmersought.com/article/69882315520/
 *
 * Matlab utility padarray symmetric. Similar approach with OpenCV PadArray.
 *
 * \param m Rows
 * \param n Columns
 */
float *padding_image(float *image, int m, int n, int patch_size) {
    float *image_padded;
    MALLOC(float, image_padded, (m + patch_size-1) * (n + patch_size-1));

    int row_pad = (patch_size-1)/2;
    int col_pad = (patch_size-1)/2;

    // copy the original image

    for(int i = 0; i < m; i++) {
        copy_row2padded(image_padded, image, i + row_pad, i, m, col_pad);
    }

    // copy boarders

    for(int i = 0; i < row_pad; i++) {
        copy_row2padded(image_padded, image, row_pad - i - 1, i, m, col_pad);
        copy_row2padded(image_padded, image, (m+row_pad) + i, m - i - 1, m, col_pad);
    }

    for(int i = 0; i < col_pad; i++) {
        int offset = n + 2*col_pad;
        copy_col2padded(image_padded, image_padded + col_pad, col_pad - i - 1, i, n + 2*row_pad, offset);
        copy_col2padded(image_padded, image_padded + col_pad, (n+col_pad) + i, n - i - 1, n + 2*row_pad, offset);
    }

    return image_padded;
}

void copy_row2padded(float *destination, float *source, int row_dest_idx, int row_source_idx, int size, int col_pad) {
    int offset = size + 2*col_pad;
    memcpy(destination + row_dest_idx * offset + col_pad, source  + row_source_idx * size, sizeof(float)*size);
}

void copy_col2padded(float *destination, float *source, int col_dest_idx, int col_source_idx, int size, int offset) {
    for(int i = 0; i < size; i++) {
        destination[i*offset + col_dest_idx] = source[i*offset + col_source_idx];
    }
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
            distance += pow(patch1[i] - patch2[i], 2); 
    }

    return sqrt(distance);
}

float apply_weighted_pixels(float *weights, float *image, int image_size) {
    float new_pixel = 0;

    for(int i = 0; i < image_size; i++) {
        new_pixel += weights[i] * image[i];
    }

    return new_pixel;
}
