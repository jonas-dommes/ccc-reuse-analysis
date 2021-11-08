#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <assert.h>

#define eps 10e-3

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {

	#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	#endif
	return result;
}

// Check result for errors, return 1 if result differs
int check_result(const float *reference, const float *result, int n) {

	for (int i = 0; i < n; i++) {
		if (abs(reference[i] - result[i]) > eps) {
			printf("Wrong result: reference[%d] = %.20f\n", i, reference[i]);
			printf("Wrong result:    result[%d] = %.20f\n",i, result[i]);
			return i;
		}
	}
	return -1;
}

void print_args(int argc, char **argv) {

	if (argc != 4) {
		printf("Error: Format should be: ./copy num_blocks threads_per_block work_per_thread \n");
		exit(1);
	} else {
		printf("num_blocks        = %d\n", atoi(argv[1]));
		printf("threads_per_block = %d\n", atoi(argv[2]));
		printf("work_per_thread   = %d\n", atoi(argv[3]));
	}
}

// Initiallize array with random float between 0 and 10
void init_random_int(int *array, int n) {

	srand(42);

	for (int i = 0; i < n; i++) {
		array[i] = (rand()/ (RAND_MAX)) * 10;
	}
}


#endif
