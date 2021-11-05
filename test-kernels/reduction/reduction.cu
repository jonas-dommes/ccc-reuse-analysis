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
#include <time.h>

#include "utility.h"

#define DEBUG 1

// Naive reduce (interleaved addressing)
__global__ void reduce1(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// strided indexing for non-divergent branching (interleaved addressing) --> Bank conflicts
__global__ void reduce2(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=1; s < blockDim.x; s *= 2)  {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// Sequential Addressing
__global__ void reduce3(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__device__ void warpReduce(volatile int* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid +  8];
	sdata[tid] += sdata[tid +  4];
	sdata[tid] += sdata[tid +  2];
	sdata[tid] += sdata[tid +  1];
}


// First Add during load and unroll last Warp
__global__ void reduce4(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s)
		sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

	if (tid < 32) warpReduce(sdata, tid);

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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
	float *h_idata = (float*) calloc(data_points, sizeof(float));
	float *h_odata = (float*) calloc(data_points, sizeof(float));

	// Initiallize input array
	init_random(h_idata, data_points);

	// // Calculate reference
	// float *reference = (float*) calloc(data_points, sizeof(float));
	//
	// clock_t begin = clock();
	//
	// for (int i = 0; i < data_points; i++) {
	// 	reference[i] = h_idata[i] * h_idata[i] - 1;
	// 	reference[i] += reference[i] * h_idata[i] - 1;
	// }
	//
	// for (int i = 0; i < 32; i++) {
	// 	reference[i] = -reference[i];
	//
	// clock_t end = clock();
	// double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	// printf("Calculated reference in %.5f ms\n", time_spent);

	// Prepare device data structures
	float *d_idata, *d_odata;
	checkCuda(cudaMalloc(&d_idata, data_points * sizeof(float)));
	checkCuda(cudaMalloc(&d_odata, data_points * sizeof(float)));
	checkCuda(cudaMemcpy(d_idata, h_idata, data_points * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(d_odata, 0, data_points * sizeof(float)));

	// // Events for timing
	// cudaEvent_t startEvent, stopEvent;
	// checkCuda(cudaEventCreate(&startEvent));
	// checkCuda(cudaEventCreate(&stopEvent));
	// float ms;

	// Run Kernel a
	// checkCuda(cudaEventRecord(startEvent, 0));
	kernel_a<<<dimGrid, dimBlock>>>(d_odata, d_idata, work_per_thread);
	checkCuda(cudaGetLastError());
	// checkCuda(cudaEventRecord(stopEvent, 0));
	// checkCuda(cudaEventSynchronize(stopEvent));
	// checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_odata, d_odata, data_points * sizeof(float), cudaMemcpyDeviceToHost));

	// Run Kernel b
	// checkCuda(cudaEventRecord(startEvent, 0));
	kernel_b<<<dimGrid, dimBlock>>>(d_odata, d_idata, work_per_thread);
	checkCuda(cudaGetLastError());
	// checkCuda(cudaEventRecord(stopEvent, 0));
	// checkCuda(cudaEventSynchronize(stopEvent));
	// checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_odata, d_odata, data_points * sizeof(float), cudaMemcpyDeviceToHost));


	// Analyse
	// int is_correct = check_result(reference, h_odata, data_points);
	// if (is_correct != -1) {
	// 	printf("Wrong result:   h_idata[%d] = %.20f\n\n",is_correct, h_idata[is_correct]);
	// } else {
	// 	printf("Correct result after %.5f ms\n", ms);
	// }

	// Cleanup
	// checkCuda(cudaEventDestroy(startEvent));
	// checkCuda(cudaEventDestroy(stopEvent));
	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));
	free(h_idata);
	free(h_odata);
	// free(reference);
}
