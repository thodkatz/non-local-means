#include "utils.cuh"

// WARNING: BLOCK_SIZE and PATCH_SIZE should be mathed with variables patch_size and blockSize (launching kernel)
// Macors used for static memory allocation
#define BLOCK_SIZE (32)
#define PATCH_SIZE (7*7)
#define SIZE (BLOCK_SIZE*PATCH_SIZE)

__global__ void filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, const int total_pixels, float *filtered_image) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int total_patch_size = patch_size * patch_size;

    // grid stride loop
    for(int pixel = tid; pixel < total_pixels; pixel+=stride) {

        // the patch that correspond to the pixel we want to filter that its patch will be compared with all the others
        __shared__ float patches_self[SIZE];
        for(int i = 0; i < total_patch_size; i++) {
            patches_self[threadIdx.x*total_patch_size + i] = patches[pixel*total_patch_size + i];
        }

        float weight = 0;
        float max = -1.0;
        float sum_weights = 0;

        // making use of register memory
        float filtered_value = 0;

        for(int i = 0; i < total_pixels/BLOCK_SIZE; i++) {

            // each thread per block copy a patch to shared memory
            __shared__ float patches_sub[SIZE];
            for(int e = 0; e < total_patch_size; e++) {
                patches_sub[threadIdx.x * total_patch_size + e] = patches[(threadIdx.x + i*BLOCK_SIZE) * total_patch_size + e];
            }
            __syncthreads();

            // each thread per block calculate the weights
            for(int j = 0; j < BLOCK_SIZE; j++) {
            weight = euclidean_distance_patch(patches_self + (threadIdx.x)*total_patch_size, patches_sub + j*total_patch_size, patch_size);
            weight = exp(-(weight*weight) / filt_sigma);

            max = (weight > max && (i*BLOCK_SIZE + j)!=pixel) ? weight : max;
            sum_weights += weight;

            float noise_pixel = *(patches_sub + j*total_patch_size + total_patch_size/2);
            filtered_value += weight * noise_pixel;
            
            }
            __syncthreads();
        }

        // neglect the weight of self distance 
        sum_weights -= 1;
        sum_weights += max;

        float noise_pixel_self = *(patches_self + threadIdx.x*total_patch_size + total_patch_size/2);
        filtered_value -= noise_pixel_self;
        filtered_value += max*noise_pixel_self;

        filtered_value /= sum_weights;
        filtered_image[pixel] = filtered_value;
    }
}

// take two patches and calculate their distance
__device__ float euclidean_distance_patch(float *patch1, float *patch2, int patch_size) {
    int total_patch_size = patch_size * patch_size;
    float distance = 0;

    for(int i = 0; i < total_patch_size; i++) {
        float temp = patch1[i] - patch2[i];
        distance += temp*temp; // kudos to student Christos Pavlidis, I didn't notice bad performance using pow()
    }
    
    return sqrt(distance);
}
