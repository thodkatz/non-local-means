#ifndef V0_H
#define V0_H

/*
 * \brief Non-local means filter CPU
 */
float *non_local_means(int m, int n, float *noise_image_array, int patch_size, float filt_sigma, float patch_sigma, int argc, char *argv[]);

void filtering(float *patches, int patch_size, float filt_sigma, float *noise_image, int total_pixels, float *filtered_image);

#endif
