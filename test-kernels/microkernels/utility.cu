#include <stdio.h>
#include <assert.h>
#include <stdlib.h>


#include "utility.cuh"

#define eps 10e-3
#define DEBUG 1

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result) {
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

void init_data_float(struct data_float* data, int datasize) {
	// Prepare host data structures
	data->h_idata = (float*) calloc(datasize, sizeof(float));
	data->h_odata = (float*) calloc(datasize, sizeof(float));

	// Initiallize input array
	init_random_float(data->h_idata, datasize);

	// Prepare device data structures
	checkCuda(cudaMalloc(&data->d_idata, datasize * sizeof(float)));
	checkCuda(cudaMalloc(&data->d_odata, datasize * sizeof(float)));
	checkCuda(cudaMemcpy(data->d_idata, data->h_idata, datasize * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(data->d_odata, 0, datasize * sizeof(float)));
}

void free_data_float(struct data_float* data) {
	free(data->h_idata);
	free(data->h_odata);
	checkCuda(cudaFree(data->d_idata));
	checkCuda(cudaFree(data->d_odata));
}

// Initiallize array with random float between 0 and 10
void init_random_float(float *array, int n) {
	srand(42);

	for (int i = 0; i < n; i++) {
		array[i] = ((float) rand()/(float) (RAND_MAX)) * 10;
	}
}

void init_data_int(struct data_int* data, int datasize) {
	// Prepare host data structures
	data->h_idata = (int*) calloc(datasize, sizeof(int));
	data->h_odata = (int*) calloc(datasize, sizeof(int));

	// Initiallize input array
	init_random_int(data->h_idata, datasize);

	// Prepare device data structures
	checkCuda(cudaMalloc(&data->d_idata, datasize * sizeof(int)));
	checkCuda(cudaMalloc(&data->d_odata, datasize * sizeof(int)));
	checkCuda(cudaMemcpy(data->d_idata, data->h_idata, datasize * sizeof(int), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(data->d_odata, 0, datasize * sizeof(int)));
}

void free_data_int(struct data_int* data) {
	free(data->h_idata);
	free(data->h_odata);
	checkCuda(cudaFree(data->d_idata));
	checkCuda(cudaFree(data->d_odata));
}

// Initiallize array with random int between 0 and 1000
void init_random_int(int *array, int n) {
	srand(42);

	for (int i = 0; i < n; i++) {
		array[i] = (int) (((float) rand()/(float) (RAND_MAX)) * 1000);
	}
}
