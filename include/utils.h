#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MALLOC(type, x, len) if((x = (type*)malloc(len * sizeof(type))) == NULL) \
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
#define CYN   "\x1B[36m"
#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

/*
 * \brief Elapsed time between two reference points using monotonic clock
 *
 * \return Elapsed time in seconds
 */
double diff_time(struct timespec, struct timespec);

void print_array(float *array, int m, int n);

void print_output_file(FILE *f, float *array, int row, int col);

#endif
