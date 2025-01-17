#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MALLOC(type, x, len)                                                                                           \
    if ((x = (type*)malloc(len * sizeof(type))) == NULL) {                                                             \
        printf("Bad alloc\n");                                                                                         \
        exit(1);                                                                                                       \
    }

#define CALLOC(type, x, len)                                                                                           \
    if ((x = (type*)calloc(len, sizeof(type))) == NULL) {                                                              \
        printf("Bad alloc\n");                                                                                         \
        exit(1);                                                                                                       \
    }
/*
 * Required to create two objects of struct timespec tic, toc
 */
#define TIC() clock_gettime(CLOCK_MONOTONIC, &tic);
#define TOC(text)                                                                                                      \
    clock_gettime(CLOCK_MONOTONIC, &toc);                                                                              \
    printf(text, diff_time(tic, toc));

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// color because why not
#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"
/*
 * \brief Elapsed time between two reference points using monotonic clock
 *
 * \return Elapsed time in seconds
 */
double diff_time(struct timespec, struct timespec);

void print_array(float* array, int m, int n);

void print_array_file(FILE* f, float* array, int row, int col);

void print_patch(float* patches, int patch_size, int pixels);

void print_patch_file(FILE* f, float* patches, int patch_size, int pixels);

/*
 * \brief Create a 2d array that each row is representing a patch in a row major format
 *
 * \param patch_size Expecting only even numbers in order for a central pixel to exist
 */
float* create_patches(float* image, int patch_size, int m, int n);

float* padding_image(float* image, int m, int n, int patch_size);

void copy_row2padded(float* destination, float* source, int row_dest_idx, int row_source_idx, int size, int col_pad);

void copy_col2padded(float* destination, float* source, int col_dest_idx, int col_source_idx, int size, int offset);

/*
 * \brief Create a Gaussian kernel symmetric
 *
 * source:
 * -
 * https://stackoverflow.com/questions/1696113/how-do-i-gaussian-blur-an-image-without-using-any-in-built-gaussian-functions
 * - https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
 * - https://stackoverflow.com/questions/54614167/trying-to-implement-gaussian-filter-in-c
 *
 * \param patch_size Expecting only even numbers in order for a central pixel to exist
 */
float* create_gauss_kernel(int patch_size, float patch_sigma);

float* euclidean_distance_matrix(float* patches, int patch_size, int rows, int cols);

float* euclidean_distance_symmetric_matrix(float* patches, int patch_size, int rows, int cols);

void euclidean_distance_matrix_per_pixel(float* weights, float* patches, int patch_size, int pixel, int cols);

float euclidean_distance_patch(float* patch1, float* patch2, int patch_size);

float apply_weighted_pixels(float* weights, float* image, int image_size);

#endif
