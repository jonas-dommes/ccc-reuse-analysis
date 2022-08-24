#include "calls_cases.cuh"
#include "utility.cuh"
#include "test_cases.cuh"

#include <stdio.h>
#include <assert.h>
#include <time.h>

#define BLOCKDIM_X 512
#define BLOCKDIM_Y 1
#define BLOCKDIM_Z 1
#define GRIDDIM_X 65535
#define GRIDDIM_Y 1
#define GRIDDIM_Z 1


int call_D1_loop_consecutive() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 8;

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
	D1_loop_consecutive<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
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

int call_D1_loop_strided() {

	int datasize = BLOCKDIM_X * GRIDDIM_X * 8;

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
	D1_loop_strided<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
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

int call_D1_if_reduce() {

	int datasize = BLOCKDIM_X * GRIDDIM_X;

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
	D1_if_reduce<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
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

int call_D1_data_dependence() {

	int datasize = BLOCKDIM_X * GRIDDIM_X;

	// Prepare Kernel dimensions
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	dim3 dimGrid(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

	struct data_int data;
	init_data_int(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_data_dependence<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(int), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_data_int(&data);

	return (int) ms;
}

int call_D1_blocks_work_same_data() {

	int datasize = BLOCKDIM_X;

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
	D1_blocks_work_same_data<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
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

int call_D1_is_prime() {

	int datasize = BLOCKDIM_X * GRIDDIM_X;

	// Prepare Kernel dimensions
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	dim3 dimGrid(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

	struct data_int data;
	init_data_int(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_is_prime<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(int), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_data_int(&data);

	return (int) ms;
}

int call_D1_gauss_sum() {

	int datasize = BLOCKDIM_X * GRIDDIM_X;

	// Prepare Kernel dimensions
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	dim3 dimGrid(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

	struct data_int data;
	init_data_int(&data, datasize);

	// Events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// Run Kernel
	checkCuda(cudaEventRecord(startEvent, 0));
	D1_gauss_sum<<<dimGrid, dimBlock>>>(data.d_idata, data.d_odata);
	checkCuda(cudaGetLastError());
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(data.h_odata, data.d_odata, datasize * sizeof(int), cudaMemcpyDeviceToHost));

	// Cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	free_data_int(&data);

	return (int) ms;
}
