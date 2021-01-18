#include <stdio.h>
#include "v0.h"
#include "utils.h"

float *non_local_means(int m, int n, float *noise_image, int patch_size, float filt_sigma, float patch_sigma) {
    float *filtered_image;
    MALLOC(float, filtered_image, m * n);

    for(int i = 0; i < m * n; i++) filtered_image[i] = 369;

    // create patches

    // apply gaussian weight
    

    return filtered_image;
}
