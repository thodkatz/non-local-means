#ifndef V1_H
#define V1_H

/*
 * \brief Non-local means filter GPU
 */
void non_local_means(float *filtered_image, int m, int n, float *noise_image_array, int patch_size, float filt_sigma, float patch_sigma, int argc, char *argv[]);

#endif
