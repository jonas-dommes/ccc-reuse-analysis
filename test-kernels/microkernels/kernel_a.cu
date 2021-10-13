/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*	notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*	notice, this list of conditions and the following disclaimer in the
*	documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*	contributors may be used to endorse or promote products derived
*	from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <assert.h>

__global__ void kernel_a(float *odata, const float *idata, int work_per_thread) {

	int offset = blockIdx.x * blockDim.x * work_per_thread + threadIdx.x;

	for (int i = 0; i < work_per_thread; i++) {
		int index = i * blockDim.x + offset;
		odata[index] = idata[index] * idata[index] - 1;
	}

	if (offset < 32) {
		odata[offset] = -odata[offset] // TODO: *(-1)?
		// Alternative: Reduce to one datapoint?
	}
}



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
		if (reference[i] != result[i]) {
			printf("Wrong result: reference[%d] = %f\t result[%d] = \n\n", i, reference[i], i, result[i]);
			return 1;
		}
	}
	return 0;
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
void init_random(float *array, int n) {

	srand(42);

	for (int i = 0; i < n; i++) {
		array[i] = ((float) rand()/(float) (RAND_MAX)) * 10;
	}
}

int main(int argc, char **argv) {

	// Handle and print arguments
	print_args(argc, argv);
	int num_blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
	int work_per_thread = atoi(argv[3]);

	int data_points = num_blocks * threads_per_block * work_per_thread;
	printf("data_points = %d\n", data_points);
	printf("num_threads = %d\n", num_blocks * threads_per_block);

	// Prepare Kernel dimensions
	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(threads_per_block, 1, 1);

	// Prepare host data structures

	// Initiallize input array

	// Prepare device data structures

	// Events for timing

	// Run Kernel

	// Analyse

	// Cleanup

}
