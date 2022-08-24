#include "calls_access_pattern.cuh"
#include "utility.cuh"
#include "test_access_pattern.cuh"

#include <stdio.h>
#include <assert.h>
#include <time.h>

#define BLOCKDIM_X 64
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 1
#define GRIDDIM_X 512
#define GRIDDIM_Y 512
#define GRIDDIM_Z 1


int call_D2_ap_transpose_block() {

	int datasize = BLOCKDIM_X * BLOCKDIM_Y;

	// Prepare Kernel dimensions
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	dim3 dimGrid(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

	struct data_float data;
	init_data_float(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D2_ap_transpose_block<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_data_float(&data);

	return (int) ms;
}

int call_D2_ap_transpose() {

	int datasize = BLOCKDIM_X * BLOCKDIM_Y * BLOCKDIM_Z * GRIDDIM_X * GRIDDIM_Y * GRIDDIM_Z;

	// Prepare Kernel dimensions
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	dim3 dimGrid(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

	struct data_float data;
	init_data_float(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D2_ap_transpose<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_data_float(&data);

	return (int) ms;
}

int call_D2_ap_stepsize() {

	int datasize = BLOCKDIM_X * BLOCKDIM_Y * BLOCKDIM_Z * GRIDDIM_X * GRIDDIM_Y * GRIDDIM_Z;

	// Prepare Kernel dimensions
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	dim3 dimGrid(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

	struct data_float data;
	init_data_float(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D2_ap_stepsize<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_data_float(&data);

	return (int) ms;
}
