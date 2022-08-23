#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "utility.cuh"

#include "test_coalescing.cuh"

#define BLOCKDIM_X 512
#define BLOCKDIM_Y 1
#define BLOCKDIM_Z 1
#define GRIDDIM_X 65535
#define GRIDDIM_Y 1
#define GRIDDIM_Z 1


int run_D1_copy_coal100() {

	int datasize = BLOCKDIM_X * GRIDDIM_X;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_copy_coal100<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_copy_coal50() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 2;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_copy_coal50<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_copy_coal25() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 4;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_copy_coal25<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_copy_coal12_5() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 8;


	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_copy_coal12_5<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_avg2_coal100() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 2;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);
	// printf("datasize: %d\n", datasize);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_avg2_coal100<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_avg2_coal50() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 2;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_avg2_coal50<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_avg4_coal25() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 4;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_avg4_coal25<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}

int run_D1_avg8_coal12_5() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 8;

	// Prepare Kernel dimensions
	dim3 dimGrid(GRIDDIM_X, 1, 1);
	dim3 dimBlock(BLOCKDIM_X, 1, 1);

	struct call_data data;
	init_call_data(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_avg8_coal12_5<<<dimGrid, dimBlock>>>(data.d_odata, data.d_idata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(float), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_call_data(&data);

	return (int) ms;
}
