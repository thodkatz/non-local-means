#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "v0.h"

int main(int argc, char *argv[]) {

    printf(CYN "VERSION 0\n" RESET);

    if(argc == 2 && strcmp(argv[1], "--debug") != 0) {
        printf("Bad arguments. If --debug option desired then:\n");
        printf("USAGE ./bin/v0 --debug\n");
        exit(-1);
    }

    FILE *noise_image_file;
    if((noise_image_file = fopen("data/noise_image_lena_const.txt", "r")) == NULL) { 
        printf("Can't open file\n");
        exit(-1);  
    }

    struct timespec tic;
    struct timespec toc;

    int m, n;
    fscanf(noise_image_file, "%d", &m);
    fscanf(noise_image_file, "%d", &n);

    printf("Reading image...\n");
    float *noise_image_array;
    MALLOC(float, noise_image_array, m * n); // row major
    for(int i = 0; i< m; i++) {
        for(int j = 0; j < n; j++) {
            if(fscanf(noise_image_file, "%f", &noise_image_array[i*n + j]) != 1)
                exit(-1);
        }
    }
    fclose(noise_image_file);

    //print_array(noise_image_array, m, n);

    float filt_sigma = 0.02;
    int patch_size = 5; // one dimension of a 2d square patch
    float patch_sigma = 5.0/3.0; // patch sigma is for the gaussian weight applied per patch. It is the standard deviation of the gaussian applied.
    assert(patch_size%2==1);

    // passing parameters to octave
    

    printf("Non-local means filtering...\n");
    float *filtered_image_array;
    TIC()
    filtered_image_array = non_local_means(m, n, noise_image_array, patch_size, filt_sigma, patch_sigma, argc, argv);
    TOC("\nTotal time elapsed filtering image: %lf\n")

    // passing parameters and output to octave
    FILE *parameters;
    parameters = fopen("data/parameters.txt", "w");
    fprintf(parameters, "%lf %d %lf", filt_sigma, patch_size, patch_sigma);
    fclose(parameters);

    printf("\nWriting output data to file...\n");
    FILE *filtered_image_file;
    filtered_image_file = fopen("data/filtered_image.txt", "w");
    print_array_file(filtered_image_file, filtered_image_array, m, n);
    fclose(filtered_image_file);

    free(filtered_image_array);
    free(noise_image_array);

    return 0;
}
