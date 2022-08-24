#include <stdio.h>

#include "test_cases.cuh"


/* - sum 8 consecutive entries to first output using loop
*  - Data to thread ratio 8;
*/
__global__ void D1_loop_consecutive(float* idata, float* odata) {

	int offset = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

	for (int i = 0; i < 8; i++) {

		odata[offset] += idata[offset + i];
	}
}

/* - sum 8 strided entries to first output using loop
*  - Data to thread ratio 8;
*/
__global__ void D1_loop_strided(float* idata, float* odata) {

	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = 0; i < 8; i++) {

		odata[offset] += idata[offset + i * stride];
	}
}

/* - Reduce 2 values additive
*  - Data to thread ratio 1;
*/
__global__ void D1_if_reduce(float* idata, float* odata) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index % 2 == 0) {
		odata[index] = idata[index] + idata[index + 1];
	}
}

/* - Data dependence
*  - Data to thread ratio 1;
*/
__global__ void D1_data_dependence(int* idata, int* odata) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	odata[idata[index] % 32]++;
}

/* - All blocks work on the same data
*  - Data to thread ratio 1 for first block only;
*/
__global__ void D1_blocks_work_same_data(float* idata, float *odata) {

	int index = threadIdx.x;

	// printf("idata: %f\n", idata[index]);

	odata[index] += idata[index];
}

/* - Compute heavy operation (is_prime?)
*  - Data to thread ratio 1;
*/
__global__ void D1_is_prime(int* idata, int *odata) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int is_prime = 1;
	int number = idata[index];

	for (int i = 0; i < number; i++) {
		if (number % i == 0) is_prime = 0;
	}

	odata[index] = is_prime;
}

/* - Compute heavy operation (gauss_sum)
*  - Data to thread ratio 1;
*/
__global__ void D1_gauss_sum(int* idata, int *odata) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int number = idata[index];
	int sum = 0;

	for (int i = 0; i <= number; i++) {
		sum += i;
	}

	odata[index] = sum;
}
