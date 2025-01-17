#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "v1.h"
#include "utils.cuh"

void non_local_means(float* filtered_image,
                     int m,
                     int n,
                     float* noise_image,
                     int patch_size,
                     float filt_sigma,
                     float patch_sigma,
                     int argc,
                     char* argv[])
{
    struct timespec tic;
    struct timespec toc;
    int total_patch_size   = patch_size * patch_size;
    const int total_pixels = m * n;

    printf("\nCreating patches...\n");
    TIC();
    float* patches;
    cudaMallocManaged(&patches, m * n * total_patch_size * sizeof(float));
    create_patches(patches, noise_image, patch_size, m, n);
    TOC("Time elapsed creating patches: %lf\n");

    float* gauss_patch;
    gauss_patch = create_gauss_kernel(patch_size, patch_sigma);

    // apply gaussian patch

    printf("\nApplying gaussian patch...\n");
    TIC();
    for (int i = 0; i < total_pixels; i++) {
        for (int k = 0; k < total_patch_size; k++) {
            patches[i * total_patch_size + k] *= gauss_patch[k];
        }
    }
    TOC("Time elapsed applying guassian patch: %lf\n");

    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, (void*)filtering, 0, 0);
    printf("Best grid size: %d\nBest block size: %d\n", gridSize, blockSize);

    gridSize  = atoi(argv[3]);
    blockSize = atoi(argv[4]);

    printf("Current blockSize: %d and gridSize: %d\n", blockSize, gridSize);
    printf("Filtering...\n");
    filtering<<<gridSize, blockSize>>>(patches, patch_size, filt_sigma, noise_image, total_pixels, filtered_image);

    cudaDeviceSynchronize();

    // debugging

    FILE* debug_filtering;
    if (argc == 6 && strcmp(argv[5], "--debug") == 0) {
        printf("Writing filtering image to file. Mode: \033[1mdebug\033[0m...\n");
        debug_filtering = fopen("data/debug/v1/filtered_image_c.txt", "w");
        print_array_file(debug_filtering, filtered_image, m, n);
        fclose(debug_filtering);
    }

    cudaFree(patches);
    free(gauss_patch);
}
