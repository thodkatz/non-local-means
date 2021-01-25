#ifndef V1_CUH
#define V1_CUH

__global__ void filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, int total_pixels, float *filtered_image);


__device__ float *euclidean_distance_matrix_per_pixel(float *patches, int patch_size, int pixel, int cols);

__device__ float euclidean_distance_patch(float *patch1, float *patch2, int pixel_size);

__device__ float apply_weighted_pixels(float *weights, float *image, int image_size);

#endif
