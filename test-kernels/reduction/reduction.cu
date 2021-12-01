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


// strided indexing for non-divergent branching (interleaved addressing) --> Bank conflicts
__global__ void reduce2(int *d_data) {

	// calc index
	unsigned int tid = threadIdx.x;
	unsigned int offset = blockIdx.x * blockDim.x;

	// do reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2)  {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			d_data[offset + index] += d_data[offset + index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0){
		d_data[blockIdx.x] = d_data[offset];
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


__device__ void warpReduce(volatile int* d_data, int index) {
	d_data[index] += d_data[index + 32];
	d_data[index] += d_data[index + 16];
	d_data[index] += d_data[index +  8];
	d_data[index] += d_data[index +  4];
	d_data[index] += d_data[index +  2];
	d_data[index] += d_data[index +  1];
}


// Unroll last Warp
__global__ void reduce4(int *d_data) {

	// calc index
	unsigned int tid = threadIdx.x;
	unsigned int offset = blockIdx.x * blockDim.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// do reduction
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			d_data[i] += d_data[i + s];
		}
		__syncthreads();
	}

	if (tid < 32) warpReduce(d_data, i);

	// write result for this block to global mem
	if (tid == 0) {
		d_data[blockIdx.x] = d_data[i];
	}
}


int main(int argc, char **argv) {

	int success = 0;
	unsigned int data_remaining = 0;

	unsigned int data_points = 2097152;
	unsigned int dimBlock = 128;

	// Get args
	if (argc == 3) {
		data_points = atoi(argv[1]);
		dimBlock = atoi(argv[2]);
	} else {
		printf("Using default datasize, argc = %d\n", argc);
	}

	// Prepare host data
	int *h_idata = (int*) calloc(data_points, sizeof(int));
	int h_odata = 0;

	// Initialize and calculate reference
	init_random_int(h_idata, data_points);
	int ref = calc_reference_reduce(h_idata, data_points);

	// Prepare device data structures
	int *d_data;
	checkCuda(cudaMalloc(&d_data, data_points * sizeof(int)));
	checkCuda(cudaMemcpy(d_data, h_idata, data_points * sizeof(int), cudaMemcpyHostToDevice));

	// ********************** reduce1 **********************
	data_remaining = data_points;

	while (data_remaining > dimBlock) { // Never ends if dimblock == 1
		printf("Call reduce1<<<%d, %d>>>\n", data_remaining / dimBlock, dimBlock);
		reduce1<<<data_remaining / dimBlock , dimBlock>>>(d_data);
		checkCuda(cudaGetLastError());

		data_remaining = data_remaining / dimBlock;
	}

	// Make last reduce (would be more efficient to do on Host)
	if (data_remaining > 1) {
		printf("Call reduce1<<<%d, %d>>> for last reduce\n", 1, data_remaining);
		reduce1<<<1 , data_remaining>>>(d_data);
		checkCuda(cudaGetLastError());
	}

	// Copy result back to host (theoretically only need first entry)
	checkCuda(cudaMemcpy(&h_odata, d_data, sizeof(int), cudaMemcpyDeviceToHost));
	// checkCuda(cudaMemcpy(h_odata, d_data, data_points * sizeof(int), cudaMemcpyDeviceToHost));

	// Compare to reference
	if (ref != h_odata) {
		printf("Reference= %d\nResult   = %d\n", ref, h_odata);
		success ++;
	} else {
		printf("## Success for reduce1!\n");
	}

	// Prepare for next Kernel
	checkCuda(cudaMemcpy(d_data, h_idata, data_points * sizeof(int), cudaMemcpyHostToDevice));
	h_odata = 0;
	// *****************************************************


	// ********************** reduce2 **********************
	data_remaining = data_points;

	while (data_remaining > dimBlock) { // Never ends if dimblock == 1
		printf("Call reduce2<<<%d, %d>>>\n", data_remaining / dimBlock, dimBlock);
		reduce2<<<data_remaining / dimBlock , dimBlock>>>(d_data);
		checkCuda(cudaGetLastError());

		data_remaining = data_remaining / dimBlock;
	}

	// Make last reduce (would be more efficient to do on Host)
	if (data_remaining > 1) {
		printf("Call reduce2<<<%d, %d>>> for last reduce\n", 1, data_remaining);
		reduce2<<<1 , data_remaining>>>(d_data);
		checkCuda(cudaGetLastError());
	}

	// Copy result back to host (theoretically only need first entry)
	checkCuda(cudaMemcpy(&h_odata, d_data, sizeof(int), cudaMemcpyDeviceToHost));
	// checkCuda(cudaMemcpy(h_odata, d_data, data_points * sizeof(int), cudaMemcpyDeviceToHost));

	// Compare to reference
	if (ref != h_odata) {
		printf("Reference= %d\nResult   = %d\n", ref, h_odata);
		success ++;
	} else {
		printf("## Success for reduce2!\n");
	}

	// Prepare for next Kernel
	checkCuda(cudaMemcpy(d_data, h_idata, data_points * sizeof(int), cudaMemcpyHostToDevice));
	h_odata = 0;
	// *****************************************************


	// ********************** reduce3 **********************
	data_remaining = data_points;

	while (data_remaining > dimBlock) { // Never ends if dimblock == 1
		printf("Call reduce3<<<%d, %d>>>\n", data_remaining / dimBlock, dimBlock);
		reduce3<<<data_remaining / dimBlock , dimBlock>>>(d_data);
		checkCuda(cudaGetLastError());

		data_remaining = data_remaining / dimBlock;
	}

	// Make last reduce (would be more efficient to do on Host)
	if (data_remaining > 1) {
		printf("Call reduce3<<<%d, %d>>> for last reduce\n", 1, data_remaining);
		reduce3<<<1 , data_remaining>>>(d_data);
		checkCuda(cudaGetLastError());
	}

	// Copy result back to host (theoretically only need first entry)
	checkCuda(cudaMemcpy(&h_odata, d_data, sizeof(int), cudaMemcpyDeviceToHost));
	// checkCuda(cudaMemcpy(h_odata, d_data, data_points * sizeof(int), cudaMemcpyDeviceToHost));

	// Compare to reference
	if (ref != h_odata) {
		printf("Reference= %d\nResult   = %d\n", ref, h_odata);
		success ++;
	} else {
		printf("## Success for reduce3!\n");
	}

	// Prepare for next Kernel
	checkCuda(cudaMemcpy(d_data, h_idata, data_points * sizeof(int), cudaMemcpyHostToDevice));
	h_odata = 0;
	// *****************************************************


	// ********************** reduce4 **********************
	data_remaining = data_points;

	while (data_remaining > dimBlock) { // Never ends if dimblock == 1
		printf("Call reduce4<<<%d, %d>>>\n", data_remaining / dimBlock, dimBlock);
		reduce4<<<data_remaining / dimBlock , dimBlock>>>(d_data);
		checkCuda(cudaGetLastError());

		data_remaining = data_remaining / dimBlock;
	}

	// Make last reduce (would be more efficient to do on Host)
	if (data_remaining > 1) {
		printf("Call reduce4<<<%d, %d>>> for last reduce\n", 1, data_remaining);
		reduce4<<<1 , data_remaining>>>(d_data);
		checkCuda(cudaGetLastError());
	}

	// Copy result back to host (theoretically only need first entry)
	checkCuda(cudaMemcpy(&h_odata, d_data, sizeof(int), cudaMemcpyDeviceToHost));
	// checkCuda(cudaMemcpy(h_odata, d_data, data_points * sizeof(int), cudaMemcpyDeviceToHost));

	// Compare to reference
	if (ref != h_odata) {
		printf("Reference= %d\nResult   = %d\n", ref, h_odata);
		success ++;
	} else {
		printf("## Success for reduce4!\n");
	}

	// Prepare for next Kernel
	checkCuda(cudaMemcpy(d_data, h_idata, data_points * sizeof(int), cudaMemcpyHostToDevice));
	h_odata = 0;
	// *****************************************************




	// Cleanup
	checkCuda(cudaFree(d_data));

	free(h_idata);

	if (success != 0) {
		printf("\nERROR: %d reductions failed!\n", success);
	}
}
