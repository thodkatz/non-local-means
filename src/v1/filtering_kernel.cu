#include "utils.cuh"

__global__ void filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, int total_pixels, float *filtered_image) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	if(tid == 2) {
		printf("Hello, I am 2\n");
	}

	for(int pixel = tid; pixel < total_pixels; pixel+=stride) {
		printf("Hello, I am %d pixel\n", pixel);
		float *weights = (float*)malloc(total_pixels * sizeof(float));
		euclidean_distance_matrix_per_pixel(weights, patches, patch_size, pixel, total_pixels);
		//weights[pixel] = pixel;
		//printf("Weight: %f \n", weights[pixel]);

        float max = -1.0;
        float sum_weights = 0;
        for(int k = 0; k<total_pixels; k++) {
            weights[k] = exp(-pow(weights[k], 2) / filt_sigma);
            if(weights[k] > max && pixel!=k) max = weights[k];
            if(pixel!=k) sum_weights += weights[k];
        }

        weights[pixel] = max;
        sum_weights += max;

        filtered_image[pixel] = apply_weighted_pixels(weights, noise_image, total_pixels);
        filtered_image[pixel] /= sum_weights;

		free(weights);
    }

}

// nearness is determined by how similar is the intensity of the pixels
__device__ void euclidean_distance_matrix_per_pixel(float *weights, float *patches, int patch_size, int pixel, int cols) {
    int total_patch_size = patch_size * patch_size;

    for(int j = 0; j < cols; j++) {
        weights[j] = euclidean_distance_patch(patches + pixel*total_patch_size, patches + j*total_patch_size, patch_size);
		printf("Weights: %f \n", weights[j]);
    }

}

// take two patches and calculate their distance
__device__ float euclidean_distance_patch(float *patch1, float *patch2, int patch_size) {
    int total_patch_size = patch_size * patch_size;

    float distance = 0;

    for(int i = 0; i < total_patch_size; i++) {
		distance += pow(patch1[i] - patch2[i], 2); 
    }
	printf("Distance %f\n", distance);
	
    return sqrt(distance);
}

__device__ float apply_weighted_pixels(float *weights, float *image, int image_size) {
    float new_pixel = 0;

    for(int i = 0; i < image_size; i++) {
        new_pixel += weights[i] * image[i];
    }

    return new_pixel;
}
