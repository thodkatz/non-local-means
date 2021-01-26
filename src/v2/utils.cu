#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "utils.cuh"

double diff_time (struct timespec start, struct timespec end) {
    uint32_t diff_sec = (end.tv_sec - start.tv_sec);
    int32_t diff_nsec = (end.tv_nsec - start.tv_nsec);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        diff_sec -= 1;
        diff_nsec = 1e9 + end.tv_nsec - start.tv_nsec;
    }

    return (1e9*diff_sec + diff_nsec)/1e9;
}

void print_array(float *array, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if(j!=col-1) printf("%0.5f ", array[i*col + j]);
            else         printf("%0.5f",  array[i*col + j]);
        }
        printf("\n");
    }
}

void print_array_file(FILE *f, float *array, int row, int col) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++){
            if(j!=col-1) fprintf(f, "%0.5f ", array[i*col + j]);
            else         fprintf(f, "%0.5f", array[i*col + j]);
        } 
        fprintf(f, "\n");
    }
}

void print_patch(float *patches, int patch_size, int pixels) {
    int total_patch_size = patch_size * patch_size;

    for(int i = 0; i < pixels; i++) {
        for(int j = 0; j < patch_size; j++) {
            for(int k = 0; k < patch_size; k++) {
                if(k!=patch_size-1) printf("%0.5f ", patches[i*total_patch_size + j*patch_size + k]);
                else                printf("%0.5f", patches[i*total_patch_size + j*patch_size + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_patch_file(FILE *f, float *patches, int patch_size, int pixels) {
    int total_patch_size = patch_size * patch_size;

    for(int i = 0; i < pixels; i++) {
        for(int j = 0; j < patch_size; j++) {
            for(int k = 0; k < patch_size; k++) {
                if(k!=patch_size-1) fprintf(f, "%0.5f ", patches[i*total_patch_size + j*patch_size + k]);
                else                fprintf(f, "%0.5f", patches[i*total_patch_size + j*patch_size + k]);
            }
            fprintf(f, "\n");
        }
        if(i!=pixels-1) fprintf(f, "\n");
    }
}

/*
 * For the pixels in the edges, patch  size extends image.
 * Fill the missing values with this flag/macro.
 *
 * TODO: how to make use of missing values in "special" patches? Wrap the image?
 */
#define OUT_OF_BOUNDS -1.0
void create_patches(float *patches, float *image, int patch_size, int m, int n) {
    int total_patch_size = patch_size * patch_size;

    // padding noise image symmetric
    float *image_padded;
    image_padded = padding_image(image, m, n, patch_size);

    int col_pad = (patch_size-1)/2;

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int top_left_corner_x = i;
            int top_left_corner_y = j;
            int patch_num = i*n + j;
            for(int patch_x = top_left_corner_x, patch_id = 0; patch_x < top_left_corner_x + patch_size; patch_x++) { 
                for(int patch_y = top_left_corner_y; patch_y < top_left_corner_y + patch_size; patch_y++) {
                    int pixel = patch_x*(n + 2*col_pad) + patch_y;
                    patches[patch_num*total_patch_size + patch_id] = image_padded[pixel];
                    patch_id++;
                }
            }
        }
    }

    free(image_padded);
}

/*
 * source: https://www.programmersought.com/article/69882315520/
 *
 * Matlab utility padarray symmetric. Similar approach with OpenCV PadArray.
 *
 * \param m Rows
 * \param n Columns
 */
float *padding_image(float *image, int m, int n, int patch_size) {
    float *image_padded;
    MALLOC(float, image_padded, (m + patch_size-1) * (n + patch_size-1));

    int row_pad = (patch_size-1)/2;
    int col_pad = (patch_size-1)/2;

    // copy the original image

    for(int i = 0; i < m; i++) {
        copy_row2padded(image_padded, image, i + row_pad, i, m, col_pad);
    }

    // copy boarders

    for(int i = 0; i < row_pad; i++) {
        copy_row2padded(image_padded, image, row_pad - i - 1, i, m, col_pad);
        copy_row2padded(image_padded, image, (m+row_pad) + i, m - i - 1, m, col_pad);
    }

    for(int i = 0; i < col_pad; i++) {
        int offset = n + 2*col_pad;
        copy_col2padded(image_padded, image_padded + col_pad, col_pad - i - 1, i, n + 2*row_pad, offset);
        copy_col2padded(image_padded, image_padded + col_pad, (n+col_pad) + i, n - i - 1, n + 2*row_pad, offset);
    }

    return image_padded;
}

void copy_row2padded(float *destination, float *source, int row_dest_idx, int row_source_idx, int size, int col_pad) {
    int offset = size + 2*col_pad;
    memcpy(destination + row_dest_idx * offset + col_pad, source  + row_source_idx * size, sizeof(float)*size);
}

void copy_col2padded(float *destination, float *source, int col_dest_idx, int col_source_idx, int size, int offset) {
    for(int i = 0; i < size; i++) {
        destination[i*offset + col_dest_idx] = source[i*offset + col_source_idx];
    }
}

float *create_gauss_kernel(int patch_size, float patch_sigma) {
    int total_patch_size = patch_size * patch_size;

    float *gauss_patch;
    MALLOC(float, gauss_patch, total_patch_size);
    //float gauss_sum = 0;
    
    for(int i = 0; i < patch_size; i++) {
        for(int j = 0; j < patch_size; j ++) {
            int x = i - (patch_size-1)/2;
            int y = j - (patch_size-1)/2;
            gauss_patch[i*patch_size + j] = exp(-(x * x + y * y) / (2 * patch_sigma * patch_sigma));
            //gauss_sum += gauss_patch[i*patch_size + j];
        }
    }

    return gauss_patch;
}
