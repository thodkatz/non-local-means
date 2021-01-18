#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "utils.h"
#include "v0.h"

int main(int argc, char *argv[]) {

    printf(CYN "<----------Version 0---------->\n" RESET);

    FILE *noise_image_file;
    if((noise_image_file = fopen("data/house_c.txt", "r")) == NULL) { 
        printf("Can't open file\n");
        exit(-1);  
    }

    int m, n;
    fscanf(noise_image_file, "%d", &m);
    fscanf(noise_image_file, "%d", &n);

    printf("Reading image...\n");
    float *noise_image_array;
    MALLOC(float, noise_image_array, m * n);
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
    float patch_sigma = 5/3;

    printf("Non-local means filtering...\n");
    float *filtered_image_array;
    filtered_image_array = non_local_means(m, n, noise_image_array, patch_size, filt_sigma, patch_sigma);

    printf("Writing output data to file...\n");
    FILE *filtered_image_file;
    filtered_image_file = fopen("data/nlm_test.txt", "w");
    print_output_file(filtered_image_file, filtered_image_array, m, n);
    fclose(filtered_image_file);

    free(filtered_image_array);
    free(noise_image_array);

    return 0;
}
