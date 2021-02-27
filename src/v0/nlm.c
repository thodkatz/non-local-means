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
            patches[i*total_patch_size + k] *= gauss_patch[k];
        }
    }
    TOC("Time elapsed applying guassian patch: %lf\n")

    float *filtered_image;
    CALLOC(float, filtered_image, total_pixels);
    printf("Filtering...\n");
    yet_another_filtering_symmetric(patches, patch_size, filt_sigma, noise_image, total_pixels, filtered_image);

    // debugging
    
    FILE *debug_patches;
    if(argc == 4 && strcmp(argv[3],"--debug") == 0) {
        printf("Writing patches to file. Mode: \033[1mdebug\033[0m...\n");
        debug_patches = fopen("data/debug/v0/patches_c.txt", "w");
        print_patch_file(debug_patches, patches, patch_size, m*n);
        fclose(debug_patches);
    }

    FILE *debug_filtering;
    if(argc == 4 && strcmp(argv[3],"--debug") == 0) {
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
        //printf("Pixel: %d\n", pixel);

        // M x N (total pixels) memory required
        float *weights;
        MALLOC(float, weights, total_pixels);
        euclidean_distance_matrix_per_pixel(weights, patches, patch_size, pixel, total_pixels);

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

void yet_another_filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, int total_pixels, float *filtered_image) {
    int total_patch_size = patch_size * patch_size;

    for(int pixel = 0; pixel < total_pixels; pixel++) {
        //printf("Pixel: %d\n", pixel);

        float weight = 0;
        float filtered_value = 0;
        
        // is it worthy to find the maximum?
#define MAXIMUM

#ifdef MAXIMUM
        float max = -1.0;
#endif
        float sum_weights = 0;

        for(int i = 0; i < total_pixels; i++) {
            weight = euclidean_distance_patch(patches + pixel*total_patch_size, patches + i*total_patch_size, patch_size);
            weight = exp(-pow(weight, 2) / filt_sigma);
#ifdef MAXIMUM
            max = (weight > max && i!=pixel) ? weight : max;
#endif
            sum_weights += weight;

            float noise_pixel = *(patches + i*total_patch_size + total_patch_size/2);
            filtered_value += weight * noise_pixel;
        }

        // neglect the weight of self distance 
#ifdef MAXIMUM
        sum_weights -= 1;
        sum_weights += max;
#endif

#ifdef MAXIMUM
        float noise_pixel = *(patches + pixel*total_patch_size + total_patch_size/2);
        filtered_value -= noise_pixel;
        filtered_value += max*noise_pixel;
#endif
        filtered_value /= sum_weights;

        filtered_image[pixel] = filtered_value;
    }
}

void yet_another_filtering_symmetric(float *patches, int patch_size, float filt_sigma, float *noise_image, int total_pixels, float *filtered_image) {
    int total_patch_size = patch_size * patch_size;

    // is it worthy to keep track of the maximum weight per pixel tho? Better results?
    float max_until_pixel[total_pixels];
    for(int i = 0; i < total_pixels; i++) max_until_pixel[i] = -1.0;
    float sum_weights_until_pixel[total_pixels]= {0};

    for(int pixel = 0; pixel < total_pixels; pixel++) {
        //printf("Pixel: %d\n", pixel);

        float weight = 0;
        float filtered_value = 0;
        float max = -1.0;
        float sum_weights = 0;

        float noise_pixel_self = *(patches + pixel*total_patch_size + total_patch_size/2);

        // calculate weights making use of the symmetry property
        
        for(int i = pixel + 1; i < total_pixels; i++) {
            weight = euclidean_distance_patch(patches + pixel*total_patch_size, patches + i*total_patch_size, patch_size);
            weight = exp(-pow(weight, 2) / filt_sigma);

            max = (weight > max) ? weight : max;
            sum_weights += weight;

            float noise_pixel = *(patches + i*total_patch_size + total_patch_size/2);
            filtered_value += weight * noise_pixel;

            /* Trying to check floating point calculation deviation with two mathematically equivalent approaches
             * Distributive property isn't valid with floating point arithmetic
             * Correct results with FLOATING_POINT defined
             */ 
#define FLOATING_POINT

#ifdef FLOATING_POINT
            filtered_image[i] += weight * noise_pixel_self;
            sum_weights_until_pixel[i] += weight;
#else
            filtered_image[i] += weight;
#endif
            max_until_pixel[i] = MAX(max_until_pixel[i], weight);
        }

        // new sum weights and max
#ifdef FLOATING_POINT
        sum_weights += sum_weights_until_pixel[pixel];
#else
        sum_weights += filtered_image[pixel];
#endif
        max = MAX(max, max_until_pixel[pixel]);

        // total filtered value per pixel
#ifdef FLOATING_POINT
        filtered_value += filtered_image[pixel];
#else
        filtered_value += filtered_image[pixel] * noise_pixel_self;
#endif

        // include the diagonal element as the max weighted value
        // how to find the max weight?
        filtered_value += max*noise_pixel_self;
        sum_weights += max;

        // normalize the filtered value
        filtered_value /= sum_weights;

        // final filtered value
        filtered_image[pixel] = filtered_value;
    }
}
