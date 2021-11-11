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
	}
}

//
// // strided indexing for non-divergent branching (interleaved addressing) --> Bank conflicts
// __global__ void reduce2(int *d_idata, int *d_odata) {
//
// 	extern __shared__ int sdata[];
//
// 	// each thread loads one element from global to shared mem
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
// 	sdata[tid] = d_idata[i];
// 	__syncthreads();
//
// 	// do reduction in shared mem
// 	for (unsigned int s=1; s < blockDim.x; s *= 2)  {
// 		int index = 2 * s * tid;
// 		if (index < blockDim.x) {
// 			sdata[index] += sdata[index + s];
// 		}
// 		__syncthreads();
// 	}
//
// 	// write result for this block to global mem
// 	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
// }
//
//
// // Sequential Addressing
// __global__ void reduce3(int *d_idata, int *d_odata) {
//
// 	extern __shared__ int sdata[];
//
// 	// each thread loads one element from global to shared mem
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
// 	sdata[tid] = d_idata[i];
// 	__syncthreads();
//
// 	// do reduction in shared mem
// 	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
// 		if (tid < s) {
// 			sdata[tid] += sdata[tid + s];
// 		}
// 		__syncthreads();
// 	}
//
// 	// write result for this block to global mem
// 	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
// }
//
//
// __device__ void warpReduce(volatile int* sdata, int tid) {
// 	sdata[tid] += sdata[tid + 32];
// 	sdata[tid] += sdata[tid + 16];
// 	sdata[tid] += sdata[tid +  8];
// 	sdata[tid] += sdata[tid +  4];
// 	sdata[tid] += sdata[tid +  2];
// 	sdata[tid] += sdata[tid +  1];
// }
//
//
// // Unroll last Warp
// __global__ void reduce4(int *d_idata, int *d_odata) {
//
// 	extern __shared__ int sdata[];
//
// 	// perform first level of reduction,
// 	// reading from global memory, writing to shared memory
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
// 	sdata[tid] = d_idata[i];
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

	int success = 0;

	// int num_kernels = 1;
	// int num_runs = 1;
	// int num_blockdims = 1;
	// // int data_points[] = {1024}; // biggest last
	// // int blockdims[] = {1024};
	// int data_points[] = {2097152}; // biggest last
	// int blockdims[] = {128};

	int num_kernels = 1;
	int num_runs = 3;
	int num_blockdims = 4;
	int data_points[] = {131072, 1048576, 2097152}; // biggest last
	int blockdims[] = {128, 256, 512, 1024};
	int ref[num_runs];

	// Prepare host data
	int *h_idata = (int*) calloc(data_points[num_runs-1], sizeof(int));
	int *h_odata = (int*) calloc(data_points[num_runs-1], sizeof(int));
	init_random_int(h_idata, data_points[num_runs-1]);



	for (size_t i = 0; i < num_runs; i++) {

		// Calculate reference
		ref[i] = calc_reference_reduce(h_idata, data_points[i]);
		printf("data_points = %d\n", data_points[i]);

		for (size_t j = 0; j < num_blockdims; j++) {

			printf("## blockdims = %d\n", blockdims[j]);

			// Prepare device data structures
			int *d_idata, *d_odata;
			checkCuda(cudaMalloc(&d_idata, data_points[i] * sizeof(int)));
			checkCuda(cudaMalloc(&d_odata, data_points[i] * sizeof(int)));
			checkCuda(cudaMemcpy(d_idata, h_idata, data_points[i] * sizeof(int), cudaMemcpyHostToDevice));
			checkCuda(cudaMemset(d_odata, 0, data_points[i] * sizeof(int)));

			// Prepare Kernel dimensions
			unsigned int num_data = data_points[i];
			unsigned int dimBlock = blockdims[j];

			while (num_data > dimBlock) { // Never ends if dimblock == 1
				printf("#### Call reduce1<<<%d, %d>>>\n", num_data / dimBlock, dimBlock);
				reduce1<<<num_data / dimBlock , dimBlock, dimBlock * sizeof(int)>>>(d_idata, d_odata);
				checkCuda(cudaMemcpy(d_idata, d_odata, data_points[i] * sizeof(int), cudaMemcpyDeviceToDevice));
				checkCuda(cudaGetLastError());
				// checkCuda(cudaDeviceSynchronize());
				num_data = num_data / dimBlock;
			}

			// Make last reduce (would be more efficient to do on Host)
			if (num_data > 1) {
				printf("#### Call reduce1<<<%d, %d>>> for last reduce\n", 1, num_data);
				reduce1<<<1 , num_data, num_data * sizeof(int)>>>(d_idata, d_odata);
				checkCuda(cudaGetLastError());
			}

			// reduce2<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			// checkCuda(cudaGetLastError());
			//
			// reduce3<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			// checkCuda(cudaGetLastError());
			//
			// reduce4<<<dimGrid, dimBlock>>>(d_idata, d_odata);
			// checkCuda(cudaGetLastError());

			checkCuda(cudaMemcpy(h_odata, d_odata, data_points[i] * sizeof(int), cudaMemcpyDeviceToHost));

			if (ref[i] != h_odata[0]) {
				printf("#### Reference= %d\nResult   = %d\n", ref[i], h_odata[0]);
				success ++;
			} else {
				printf("## Success for blockdims = %d\n", blockdims[j]);
			}

			memset(h_odata, 0, data_points[i] * sizeof(int));

			checkCuda(cudaFree(d_idata));
			checkCuda(cudaFree(d_odata));


		}


		printf("Finished for data_points = %d\n", data_points[i]);
	}
	free(h_idata);
	free(h_odata);

	if (success != 0) {
		printf("\nERROR: %d reductions failed!\n", success);
	}
}
