#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MALLOC(type, x, len) if((x = (type*)malloc(len * sizeof(type))) == NULL) \
                                {printf("Bad alloc\n"); exit(1);}

#define CALLOC(type, x, len) if((x = (type*)calloc(len, sizeof(type))) == NULL) \
                                {printf("Bad alloc\n"); exit(1);}
/*
 * Required to create two objects of struct timespec tic, toc
 */
#define TIC() clock_gettime(CLOCK_MONOTONIC, &tic);
#define TOC(text) clock_gettime(CLOCK_MONOTONIC, &toc); \
                printf(text, diff_time(tic,toc));

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

void print_array(float *array, int m, int n);

void print_array_file(FILE *f, float *array, int row, int col);

void print_patch(float *patches, int patch_size, int pixels);

void print_patch_file(FILE *f, float *patches, int patch_size, int pixels);

#endif
