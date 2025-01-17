#include "utils.cuh"

__global__ void yet_another_filtering(float* patches,
                                      int patch_size,
                                      float filt_sigma,
                                      float* noise_image,
                                      const int total_pixels,
                                      float* filtered_image);

__global__ void filtering(
    float* patches, int patch_size, float filt_sigma, float* noise_image, const int total_pixels, float* filtered_image)
{

    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int total_patch_size = patch_size * patch_size;

    // grid stride loop
    for (int pixel = tid; pixel < total_pixels; pixel += stride) {

        float weight         = 0;
        float filtered_value = 0;
        float max            = -1.0;
        float sum_weights    = 0;

        for (int i = 0; i < total_pixels; i++) {
            weight = euclidean_distance_patch(
                patches + pixel * total_patch_size, patches + i * total_patch_size, patch_size);
            weight = exp(-weight * weight / filt_sigma);

            max = (weight > max && i != pixel) ? weight : max;
            sum_weights += weight;

            float noise_pixel = *(patches + i * total_patch_size + total_patch_size / 2);
            filtered_value += weight * noise_pixel;
        }

        // neglect the weight of self distance
        sum_weights -= 1;
        sum_weights += max;

        float noise_pixel_self = *(patches + pixel * total_patch_size + total_patch_size / 2);
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
        float temp = patch1[i] - patch2[i]; // kudos to christos pavlidis, I didn't notice bad performance using pow()
        distance += temp * temp;
    }

    return sqrt(distance);
}

// making use of the symmetric property of the weights requires thread communication. The disrupted view of the weights
// and the need of global writes will result eventually to poor performance 
// WARNING: the following function doesn't work
__global__ void yet_another_filtering(
    float* patches, int patch_size, float filt_sigma, float* noise_image, const int total_pixels, float* filtered_image)
{

    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int total_patch_size = patch_size * patch_size;

    // grid stride loop
    for (int pixel = tid; pixel < total_pixels; pixel += stride) {

        float weight         = 0;
        float filtered_value = 0;
        float max            = -1.0;
        float sum_weights    = 0;

        float noise_pixel_self = *(patches + pixel * total_patch_size + total_patch_size / 2);

        for (int i = pixel + 1; i < total_pixels; i++) {
            weight = euclidean_distance_patch(
                patches + pixel * total_patch_size, patches + i * total_patch_size, patch_size);
            weight = exp(-weight * weight / filt_sigma);

            max = (weight > max) ? weight : max;
            sum_weights += weight;

            float noise_pixel = *(patches + i * total_patch_size + total_patch_size / 2);
            filtered_value += weight * noise_pixel;

            // need atomic here
            filtered_image[i] += weight;
        }

        // total filtered value per pixel
        // block for 0...pixel
        // to block all the threads the kernel should be completed
        // we could stop kernel here and do the multiplication in cpu
        filtered_value += filtered_image[pixel] * noise_pixel_self;

        // include the diagonal element as the max weighted value
        // WARNING: max here doesnt have global view due to making use of symmetry by thread intercommunication
        filtered_value += max * noise_pixel_self;
        sum_weights += filtered_image[pixel];
        sum_weights += max;

        // normalize the filtered value
        filtered_value /= sum_weights;

        // final filtered value
        filtered_image[pixel] = filtered_value;
    }
}
