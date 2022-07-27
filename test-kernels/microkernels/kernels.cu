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

/*
clang++-9 -S -emit-llvm -O1 ~/ba/code/ccc-reuse-analysis/test-kernels/microkernels/kernels.cu --cuda-gpu-arch=sm_75 -fno-discard-value-names --cuda-device-only -o ~/ba/code/ccc-reuse-analysis/test-kernels/microkernels/kernels.ll
*/

#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "utility.h"

#define DEBUG 1
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void TEST_access_pattern(float *odata) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	odata[0] = odata[i];
	odata[7] = odata[i];
}

// TEST KERNEL
__global__ void TEST_nested_conditions(float *odata) {

	int i = (int) odata[7];

	if (i < 1000) {

		i =  (int) odata[i];

	}
	if (i < 32) {

		odata[i] = 10;
	} else {

		odata[i] *= -10;

	}
}

__global__ void TEST_for_loops(float *odata, const float *idata) {

	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < 32; i++) {
		int index = i * blockDim.x + offset;
		odata[index] = idata[index] * idata[index] - 1;
	}

	for (int j = 0; j < 32; j++) {
		int index = j * blockDim.x + offset;
		odata[index] = idata[index] * idata[index] - 1;
	}

	if (offset < 32) {
		odata[offset] = -odata[offset];
	}
}

__global__ void TEST_nested_for_loops(float *odata, const float *idata) {

	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < 32; i++) {
		int index = i * blockDim.x + offset;
		odata[index] = idata[index] * idata[index] - 1;


		for (int j = 0; j < 32; j++) {
			int index = j * blockDim.x + offset;
			odata[index] = idata[index] * idata[index] - 1;
		}
	}

	if (offset < 32) {
		odata[offset] = -odata[offset];
	}
}

// Data reuse of first few entries
__global__ void kernel_a(float *odata, const float *idata, int work_per_thread) {

	int offset = blockIdx.x * blockDim.x * work_per_thread + threadIdx.x;

	for (int i = 0; i < work_per_thread; i++) {
		int index = i * blockDim.x + offset;
		odata[index] = idata[index] * idata[index] - 1;
	}

	if (offset < 32) {
		odata[offset] = -odata[offset];
	}
}

// Using data multiple times
__global__ void kernel_b(float *odata, const float *idata, int work_per_thread) {

	int offset = blockIdx.x * blockDim.x * work_per_thread + threadIdx.x;

	for (int i = 0; i < work_per_thread; i++) {
		int index = i * blockDim.x + offset;
		odata[index] = idata[index] * idata[index] - 1;
		odata[index] += odata[index] * idata[index] - 1;
	}

	if (offset < 32) {
		odata[offset] = -odata[offset];
	}
}

// Naive reduce (interleaved addressing)
__global__ void reduce1(int *d_data) {

	// calc index
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// do reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			d_data[i] = d_data[i] + d_data[i + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		d_data[blockIdx.x] = d_data[i];
	}
}

// Sequential Addressing
__global__ void reduce3(int *d_data) {

	// calc index
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// do reduction
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			d_data[i] += d_data[i + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		d_data[blockIdx.x] = d_data[i];
	}
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
__global__ void copy(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

// 2d simple transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeSimple(float *odata, const float *idata)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.y;
	int indexIn = row * width + col;
	// indexin = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x
	int indexOut = col * height + row;

	odata[indexOut]  = idata[indexIn];

}


int main(int argc, char **argv) {
	//
	// // Handle and print arguments
	// print_args(argc, argv);
	// int num_blocks = atoi(argv[1]);
	// int threads_per_block = atoi(argv[2]);
	// int work_per_thread = atoi(argv[3]);
	//
	// int data_points = num_blocks * threads_per_block * work_per_thread;
	// printf("data_points = %d\n", data_points);
	// printf("num_threads = %d\n", num_blocks * threads_per_block);
	//
	// // Prepare Kernel dimensions
	// dim3 dimGrid(num_blocks, 1, 1);
	// dim3 dimBlock(threads_per_block, 1, 1);
	//
	// // Prepare host data structures
	// float *h_idata = (float*) calloc(data_points, sizeof(float));
	// float *h_odata = (float*) calloc(data_points, sizeof(float));
	//
	// // Initiallize input array
	// init_random(h_idata, data_points);
	//
	// // // Calculate reference
	// // float *reference = (float*) calloc(data_points, sizeof(float));
	// //
	// // clock_t begin = clock();
	// //
	// // for (int i = 0; i < data_points; i++) {
	// // 	reference[i] = h_idata[i] * h_idata[i] - 1;
	// // 	reference[i] += reference[i] * h_idata[i] - 1;
	// // }
	// //
	// // for (int i = 0; i < 32; i++) {
	// // 	reference[i] = -reference[i];
	// //
	// // clock_t end = clock();
	// // double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	// // printf("Calculated reference in %.5f ms\n", time_spent);
	//
	// // Prepare device data structures
	// float *d_idata, *d_odata;
	// checkCuda(cudaMalloc(&d_idata, data_points * sizeof(float)));
	// checkCuda(cudaMalloc(&d_odata, data_points * sizeof(float)));
	// checkCuda(cudaMemcpy(d_idata, h_idata, data_points * sizeof(float), cudaMemcpyHostToDevice));
	// checkCuda(cudaMemset(d_odata, 0, data_points * sizeof(float)));
	//
	// // // Events for timing
	// // cudaEvent_t startEvent, stopEvent;
	// // checkCuda(cudaEventCreate(&startEvent));
	// // checkCuda(cudaEventCreate(&stopEvent));
	// // float ms;
	//
	// // Run Kernel a
	// // checkCuda(cudaEventRecord(startEvent, 0));
	// kernel_a<<<dimGrid, dimBlock>>>(d_odata, d_idata, work_per_thread);
	// checkCuda(cudaGetLastError());
	// // checkCuda(cudaEventRecord(stopEvent, 0));
	// // checkCuda(cudaEventSynchronize(stopEvent));
	// // checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	// checkCuda(cudaMemcpy(h_odata, d_odata, data_points * sizeof(float), cudaMemcpyDeviceToHost));
	//
	// // Run Kernel b
	// // checkCuda(cudaEventRecord(startEvent, 0));
	// kernel_b<<<dimGrid, dimBlock>>>(d_odata, d_idata, work_per_thread);
	// checkCuda(cudaGetLastError());
	// // checkCuda(cudaEventRecord(stopEvent, 0));
	// // checkCuda(cudaEventSynchronize(stopEvent));
	// // checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	// checkCuda(cudaMemcpy(h_odata, d_odata, data_points * sizeof(float), cudaMemcpyDeviceToHost));
	//
	//
	// // Analyse
	// // int is_correct = check_result(reference, h_odata, data_points);
	// // if (is_correct != -1) {
	// // 	printf("Wrong result:   h_idata[%d] = %.20f\n\n",is_correct, h_idata[is_correct]);
	// // } else {
	// // 	printf("Correct result after %.5f ms\n", ms);
	// // }
	//
	// // Cleanup
	// // checkCuda(cudaEventDestroy(startEvent));
	// // checkCuda(cudaEventDestroy(stopEvent));
	// checkCuda(cudaFree(d_idata));
	// checkCuda(cudaFree(d_odata));
	// free(h_idata);
	// free(h_odata);
	// free(reference);
}
