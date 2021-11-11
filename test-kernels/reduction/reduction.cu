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
__global__ void reduce1(int *d_idata, int *d_odata) {

	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = d_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)  {
		d_odata[blockIdx.x] = sdata[0];

		// Recursively call kernel
		if (blockIdx.x == 0) {
			if (gridDim.x >= blockDim.x) {
				reduce1<<<gridDim.x / blockDim.x, blockdim.x>>>(d_idata, d_odata);
			} else if (gridDim.x > 1) {
				reduce1<<<1, gridDim.x>>>(d_idata, d_odata);
			}
		}
	}


// strided indexing for non-divergent branching (interleaved addressing) --> Bank conflicts
__global__ void reduce2(int *d_idata, int *d_odata) {

	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = d_idata[i];
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
	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}


// Sequential Addressing
__global__ void reduce3(int *d_idata, int *d_odata) {

	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = d_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}


__device__ void warpReduce(volatile int* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid +  8];
	sdata[tid] += sdata[tid +  4];
	sdata[tid] += sdata[tid +  2];
	sdata[tid] += sdata[tid +  1];
}


// Unroll last Warp
__global__ void reduce4(int *d_idata, int *d_odata) {

	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = d_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s)
		sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

	if (tid < 32) warpReduce(sdata, tid);

	// write result for this block to global mem
	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

// // First Add during load and unroll last Warp
// __global__ void reduce4(int *d_idata, int *d_odata) {
//
// 	extern __shared__ int sdata[];
//
// 	// perform first level of reduction,
// 	// reading from global memory, writing to shared memory
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
// 	sdata[tid] = d_idata[i] + d_idata[i+blockDim.x];
// 	__syncthreads();
//
// 	// do reduction in shared mem
// 	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
// 		if (tid < s)
// 		sdata[tid] += sdata[tid + s];
// 		__syncthreads();
// 	}
//
// 	if (tid < 32) warpReduce(sdata, tid);
//
// 	// write result for this block to global mem
// 	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
// }


int main(int argc, char **argv) {

	int num_kernels = 1;
	int num_runs = 1;
	int num_blockdims = 1;
	int data_points[num_runs] = {1024}; // biggest last
	int blockdims[num_blockdims] = {1024};
	// int data_points[num_runs] = {2097152}; // biggest last
	// int blockdims[num_blockdims] = {128};

	// int num_kernels = 4;
	// int num_runs = 3;
	// int num_blockdims = 4;
	// int data_points[num_runs] = {131072, 1048576, 2097152}; // biggest last
	// int blockdims[num_blockdims] = {128, 356, 512, 1024};

	// Prepare host data
	int *h_idata = (int*) calloc(data_points[num_runs-1], sizeof(int));
	int *h_odata = (int*) calloc(data_points[num_runs-1], sizeof(int));
	init_random_int(h_idata, data_points[num_runs-1]);

	// Calculate reference
	int sum = calc_reference_reduce(h_idata, data);

	for (size_t i = 0; i < num_runs; i++) {
		for (size_t j = 0; j < num_blockdims; j++) {

			// Prepare Kernel dimensions
			dim3 dimGrid(data_points[i] / blockdims[j], 1, 1);
			dim3 dimBlock(blockdims[j], 1, 1);

			// Prepare device data structures
			float *d_idata, *d_odata;
			checkCuda(cudaMalloc(&d_idata, data_points[i] * sizeof(int)));
			checkCuda(cudaMalloc(&d_odata, data_points[i] * sizeof(int)));
			checkCuda(cudaMemcpy(d_idata, h_idata, data_points[i] * sizeof(int), cudaMemcpyHostToDevice));
			checkCuda(cudaMemset(d_odata, 0, data_points[i] * sizeof(int)));

			reduce1<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			checkCuda(cudaGetLastError());


			// reduce2<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			// checkCuda(cudaGetLastError());
			//
			// reduce3<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			// checkCuda(cudaGetLastError());
			//
			// reduce4<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			// checkCuda(cudaGetLastError());


			checkCuda(cudaMemcpy(h_odata, d_odata, data_points[i] * sizeof(int), cudaMemcpyDeviceToHost));

			if (sum != h_odata[0]) {
				printf("Reference= %d\nResult   = %d", sum, h_odata[0]);
			}

			checkCuda(cudaFree(d_idata));
			checkCuda(cudaFree(d_odata));
		}

		// memset(h_odata, 0, data_points[i] * sizeof(int));
	}
	free(h_idata);
	free(h_odata);
}
