#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

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
            printf("%f ", array[i*col + j]);
        }
        printf("\n");
    }
}

void print_output_file(FILE *f, float *array, int row, int col) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++){
            fprintf(f, "%f ", array[i*col + j]);
        } 
        fprintf(f, "\n");
    }
}
