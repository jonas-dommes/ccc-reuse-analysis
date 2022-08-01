#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <assert.h>

#define eps 10e-3

inline cudaError_t checkCuda(cudaError_t result);

int check_result(const float *reference, const float *result, int n);

void print_args(int argc, char **argv);

void init_random(float *array, int n);

#endif
