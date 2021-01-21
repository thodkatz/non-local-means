#ifndef V0_H
#define V0_H

/*
 * \brief Non-local means filter CPU
 */
float *non_local_means(int m, int n, float *noise_image_array, int patch_size, float filt_sigma, float patch_sigma);

/*
 * \brief Create a 2d array that each row is representing a patch in a row major format
 *
 * \param patch_size Expecting only even numbers in order for a central pixel to exist
 */
float *create_patches(float *image, int patch_size, int m, int n);

/*
 * \brief Create a Gaussian kernel symmetric
 *
 * source:
 * - https://stackoverflow.com/questions/1696113/how-do-i-gaussian-blur-an-image-without-using-any-in-built-gaussian-functions
 * - https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
 * - https://stackoverflow.com/questions/54614167/trying-to-implement-gaussian-filter-in-c
 *
 * \param patch_size Expecting only even numbers in order for a central pixel to exist
 */
float *create_gauss_kernel(int patch_size, float patch_sigma);

float *euclidean_distance_matrix(float *patches, int patch_size, int m, int n);

float euclidean_distance_patch(float *patch1, float *patch2, int patch_size);

float apply_weighted_pixels(float *weights, float *image, int image_size);
#endif
