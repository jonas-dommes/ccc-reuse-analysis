#include <stdio.h>

#include "test_access_pattern.cuh"


/* - Transpose 2d Matrix in single Block
*  - Data to thread ratio 1;
*/
__global__ void D2_ap_transpose_block(float* idata, float *odata) {

	int i_in = threadIdx.x + threadIdx.y * blockDim.x;
	int i_out = threadIdx.y + threadIdx.x * blockDim.y;

	odata[i_out] = idata[i_in];
}

/* - Transpose 2d Matrix
*  - Data to thread ratio 1;
*/
__global__ void D2_ap_transpose(float* idata, float *odata) {

	int width_in = blockDim.x * gridDim.x;
	int row_in = blockIdx.y * blockDim.y + threadIdx.y;
	int x_in = blockIdx.x * blockDim.x + threadIdx.x;
	int i_in = width_in * row_in + x_in;

	int width_out = blockDim.y * gridDim.y;
	int row_out = blockIdx.x * blockDim.x + threadIdx.x;
	int x_out = blockIdx.y * blockDim.y + threadIdx.y;
	int i_out = width_out * row_out + x_out;

	odata[i_out] = idata[i_in];
}

/* - Random Program to test stepsize
*  - Data to thread ratio 1;
*/
__global__ void D2_ap_stepsize(float* idata, float *odata) {

	int i_in = threadIdx.y * blockDim.x + threadIdx.x;
	int i_out = threadIdx.x * blockDim.y + threadIdx.y;

	odata[i_out] = idata[i_in];
}
