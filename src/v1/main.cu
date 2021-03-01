#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "v1.h"
#include "utils.cuh"

int main(int argc, char* argv[])
{

    printf(CYN "VERSION 1\n" RESET);

    if (argc < 5) {
        printf("Bad arguments\n");
        printf("USAGE ./bin/v1 <noised_image.txt> <patch size> <grid size> <block size> --debug\n");
        printf("--debug is optional\n");
        exit(-1);
    }

    FILE* noise_image_file;
    if ((noise_image_file = fopen(argv[1], "r")) == NULL) {
        printf("Can't open file\n");
        exit(-1);
    }

    int patch_size    = atoi(argv[2]);
    int block_size    = atoi(argv[3]);
    int threads       = atoi(argv[4]);
    assert(patch_size % 2 == 1);

    struct timespec tic;
    struct timespec toc;

    int m, n;
    fscanf(noise_image_file, "%d", &m);
    fscanf(noise_image_file, "%d", &n);

    printf("Reading image...\n");
    float* noise_image_array;
    MALLOC(float, noise_image_array, m* n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(noise_image_file, "%f", &noise_image_array[i * n + j]) != 1)
                exit(-1);
        }
    }
    fclose(noise_image_file);

    // print_array(noise_image_array, m, n);

    float filt_sigma  = 0.02;
    float patch_sigma = 5.0 / 3.0; 

    printf("Non-local means filtering...\n");
    float* filtered_image_array;
    cudaMallocManaged(&filtered_image_array, m * n * sizeof(float));
    TIC()
    non_local_means(filtered_image_array, m, n, noise_image_array, patch_size, filt_sigma, patch_sigma, argc, argv);
    TOC("\nTotal time elapsed filtering image: %lf\n")

    /* ------------------------------ Save results ------------------------------ */

    char* resultsPath = "./results/results.csv";
    FILE* fp;

    if ((fp = fopen(resultsPath, "a+")) == NULL) {
        printf("File does not exist.\nExiting...");
        exit(1);
    }

    fprintf(fp, "%s,%s,%d,%d,%d,%lf\n", argv[0], argv[1], patch_size, block_size, threads, diff_time(tic, toc));

    /* ---------------------------------- Other --------------------------------- */

    printf("\nWriting output data to file...\n");
    FILE* filtered_image_file;
    filtered_image_file = fopen("data/filtered_image_v1.txt", "w");
    print_array_file(filtered_image_file, filtered_image_array, m, n);
    fclose(filtered_image_file);

    cudaFree(filtered_image_array);
    free(noise_image_array);

    return 0;
}
