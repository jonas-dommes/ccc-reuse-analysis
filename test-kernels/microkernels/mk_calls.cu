#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "utility.cuh"

#include "copy.cuh"

#define BLOCKDIM_X 1024
#define BLOCKDIM_Y 1
#define BLOCKDIM_Z 1
#define GRIDDIM_X 65535
#define GRIDDIM_Y 1
#define GRIDDIM_Z 1


int run_copy1D_coal100() {

	int datasize = BLOCKDIM_X * GRIDDIM_X;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	// Prepare host data structures
	float *h_idata = (float*) calloc(datasize, sizeof(float));
	float *h_odata = (float*) calloc(datasize, sizeof(float));

	// Initiallize input array
	init_random(h_idata, datasize);

	// Prepare device data structures
	float *d_idata, *d_odata;
	checkCuda(cudaMalloc(&d_idata, datasize * sizeof(float)));
	checkCuda(cudaMalloc(&d_odata, datasize * sizeof(float)));
	checkCuda(cudaMemcpy(d_idata, h_idata, datasize * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(d_odata, 0, datasize * sizeof(float)));

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	copy1D_coal100<<<dimGrid, dimBlock>>>(d_odata, d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_odata, d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	return (int) ms;
}
