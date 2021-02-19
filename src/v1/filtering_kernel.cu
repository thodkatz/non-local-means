#include "utils.cuh"

__global__ void filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, const int total_pixels, float *filtered_image) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int total_patch_size = patch_size * patch_size;
    // grid stride loop
    for(int pixel = tid; tid < total_pixels; tid+=stride) {

        float weight = 0;
        float filtered_value = 0;
        float max = -1.0;
        float sum_weights = 0;

        for(int i = 0; i < total_pixels; i++) {
            weight = euclidean_distance_patch(patches + pixel*total_patch_size, patches + i*total_patch_size, patch_size);
            weight = exp(-pow(weight, 2) / filt_sigma);

            max = (weight > max && i!=pixel) ? weight : max;
            sum_weights += weight;

            filtered_value += weight * noise_image[i];
        }

        // neglect the weight of self distance 
        sum_weights -= 1;
        sum_weights += max;

        float noise_pixel = *(patch_size + pixel*total_patch_size + total_patch_size/2);
        filtered_value -= noise_pixel;
        filtered_value += max*noise_pixel;
        filtered_value /= sum_weights;

        filtered_image[pixel] = filtered_value;
    }
}

// take two patches and calculate their distance
__device__ float euclidean_distance_patch(float *patch1, float *patch2, int patch_size) {
    int total_patch_size = patch_size * patch_size;
    float distance = 0;

    for(int i = 0; i < total_patch_size; i++) {
        distance += pow(patch1[i] - patch2[i], 2); 
    }
    
    return sqrt(distance);
}
