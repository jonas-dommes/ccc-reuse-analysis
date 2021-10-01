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

#define NUM_REPS 1

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


// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms) {
	bool passed = true;
	for (int i = 0; i < n; i++) {
		if (res[i] != ref[i]) {
			printf("%d %f %f\n", i, res[i], ref[i]);
			printf("%25s\n", "*** FAILED ***");
			passed = false;
			break;
		}
	}
	if (passed) {
		printf("Runtime: %6.3f ms\t", ms );
		printf("Bandwidth: %8.2f GB/s\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms );
	}
}

#define PRINT_IDX 132
/* Kernel to copy 1 dimensional array
*  int work_per_thread  number of datapoints each thread works
*/
__global__ void copy_array(float *odata, const float *idata, int work_per_thread) {

	int tid = blockIdx.x * blockDim.x * work_per_thread + threadIdx.x;
	// printf("%2d tid: %d\n", blockIdx.x, tid);

	// if(tid == PRINT_IDX) printf("blockIdx.x=%d   blockDim.x=%d   gridDim.x=%d   threadIdx.x=%d \n", blockIdx.x, blockDim.x, gridDim.x, threadIdx.x);

	// if(tid == PRINT_IDX) printf("idata[%d]: %f\n", PRINT_IDX, idata[PRINT_IDX]);

	for (int i = 0; i < work_per_thread; i++) {
		int index = i * blockDim.x + tid;
		odata[index] = idata[index];

		// if(tid == PRINT_IDX) printf("i: %d   index: %d\n", i, index);

	}

	// if(tid == PRINT_IDX) printf("odata[%d]: %f\n", PRINT_IDX, odata[PRINT_IDX]);
}

__global__ void memcpyByCols(float *idata, float *odata, unsigned int size) {
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int idx = 0; idx < size; idx++) {
		odata[globalId * size + idx] = idata[globalId * size + idx];
	}
}

__global__ void memcpyByRows(float *idata, float *odata, unsigned int size) {
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int idx = 0; idx < size; idx++) {
		odata[idx * size + globalId] = idata[idx * size + globalId];
	}
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

int main(int argc, char **argv) {

	// Handle and print arguments
	print_args(argc, argv);
	int num_blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
	int work_per_thread = atoi(argv[3]);

	int data_size = num_blocks * threads_per_block * work_per_thread;
	printf("data_size = %d\n", data_size);
	printf("num_threads = %d\n", num_blocks * threads_per_block);

	// Prepare Kernel dimensions
	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(threads_per_block, 1, 1);

	// Prepare host data
	float *h_idata = (float*) calloc(data_size, sizeof(float));
	float *h_odata = (float*) calloc(data_size, sizeof(float));
	float *original_idata = (float*) calloc(data_size, sizeof(float));

	srand(42);

	for (int i = 0; i < data_size; i++) {
		h_idata[i] = ((float) rand()/(float) (RAND_MAX)) * 10;
		original_idata[i] = h_idata[i];
	}

	// Prepare device data
	float *d_idata, *d_odata;
	checkCuda(cudaMalloc(&d_idata, data_size * sizeof(float)));
	checkCuda(cudaMalloc(&d_odata, data_size * sizeof(float)));
	checkCuda(cudaMemcpy(d_idata, h_idata, data_size * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(d_odata, 0, data_size * sizeof(float)));

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	copy_array<<<dimGrid, dimBlock>>>(d_odata, d_idata, work_per_thread);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_odata, d_odata, data_size * sizeof(float), cudaMemcpyDeviceToHost));

	// Analyse
	postprocess(original_idata, h_odata, data_size, ms);

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));
	free(h_idata);
	free(h_odata);
	free(original_idata);
}
