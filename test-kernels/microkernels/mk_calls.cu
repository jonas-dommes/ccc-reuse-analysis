#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "utility.h"

#include "microkernels.cuh"


void run_kernel_a(int argc, char **argv) {

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

	// // Run Kernel b
	// // checkCuda(cudaEventRecord(startEvent, 0));
	// kernel_b<<<dimGrid, dimBlock>>>(d_odata, d_idata, work_per_thread);
	// checkCuda(cudaGetLastError());
	// // checkCuda(cudaEventRecord(stopEvent, 0));
	// // checkCuda(cudaEventSynchronize(stopEvent));
	// // checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	// checkCuda(cudaMemcpy(h_odata, d_odata, data_points * sizeof(float), cudaMemcpyDeviceToHost));


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
