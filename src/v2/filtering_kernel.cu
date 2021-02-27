#include "utils.cuh"

__global__ void filtering(
    float* patches, int patch_size, float filt_sigma, float* noise_image, const int total_pixels, float* filtered_image)
{

    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int total_patch_size = patch_size * patch_size;

    extern __shared__ float s[];
    float* patches_self = s;
    float* patches_sub  = (float*)&patches_self[blockDim.x * total_patch_size];

    // grid stride loop
    for (int pixel = tid; pixel < total_pixels; pixel += stride) {

        // the patch that correspond to the pixel we want to filter that its patch will be compared with all the others
        for (int i = 0; i < total_patch_size; i++) {
            patches_self[threadIdx.x * total_patch_size + i] = patches[pixel * total_patch_size + i];
        }
        //__syncthreads();

        float weight      = 0;
        float max         = -1.0;
        float sum_weights = 0;

        // making use of register memory
        float filtered_value = 0;

        for (int i = 0; i < total_pixels / blockDim.x; i++) {

            // each thread per block copy a patch to shared memory
            for (int e = 0; e < total_patch_size; e++) {
                patches_sub[threadIdx.x * total_patch_size + e] =
                    patches[(threadIdx.x + i * blockDim.x) * total_patch_size + e];
            }
            __syncthreads();

            // each thread per block calculate the weights
            for (int j = 0; j < blockDim.x; j++) {
                weight = euclidean_distance_patch(
                    patches_self + (threadIdx.x) * total_patch_size, patches_sub + j * total_patch_size, patch_size);
                weight = exp(-(weight * weight) / filt_sigma);

                max = (weight > max && (i * blockDim.x + j) != pixel) ? weight : max;
                sum_weights += weight;

                float noise_pixel = *(patches_sub + j * total_patch_size + total_patch_size / 2);
                filtered_value += weight * noise_pixel;
            }
            __syncthreads();
        }

        // neglect the weight of self distance
        sum_weights -= 1;
        sum_weights += max;

        float noise_pixel_self = *(patches_self + threadIdx.x * total_patch_size + total_patch_size / 2);
        filtered_value -= noise_pixel_self;
        filtered_value += max * noise_pixel_self;

        filtered_value /= sum_weights;
        filtered_image[pixel] = filtered_value;
    }
}

// take two patches and calculate their distance
__device__ float euclidean_distance_patch(float* patch1, float* patch2, int patch_size)
{
    int total_patch_size = patch_size * patch_size;
    float distance       = 0;

    for (int i = 0; i < total_patch_size; i++) {
        float temp = patch1[i] - patch2[i];
        distance += temp * temp; // kudos to student Christos Pavlidis, I didn't notice bad performance using pow()
    }

    return sqrt(distance);
}
