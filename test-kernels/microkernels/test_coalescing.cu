#include <stdio.h>

#include "test_coalescing.cuh"

/*  Simple 1D copy kernel, every called thread copies one value
*  - 100% coalesced
*  - Data to thread ratio 1;
*/
__global__ void D1_copy_coal100(float* idata, float *odata) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	odata[index] = idata[index];
}

/*  Simple 1D copy kernel of every second value, every called thread copies one value
*  - 50% coalesced
*  - Data to thread ratio 2;
*/
__global__ void D1_copy_coal50(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

	odata[index] = idata[index];
}

/*  Simple 1D copy kernel of every fourth value, every called thread copies one value
*  - 25% coalesced
*  - Data to thread ratio 4;
*/
__global__ void D1_copy_coal25(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

	odata[index] = idata[index];
}

/*  Simple 1D copy kernel of every eighth value, every called thread copies one value
*  - 12.5% coalesced
*  - Data to thread ratio 8;
*/
__global__ void D1_copy_coal12_5(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

	odata[index] = idata[index];
}

/*  Averages two strided values and writes them back
*  - 100% coalesced
*  - Data to thread ratio 2;
*/
__global__ void D1_avg2_coal100(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x);
	int stride = blockDim.x * gridDim.x;

	float avg = idata[index];
	avg += idata[index + stride];
	avg /= 2.;

	odata[index] = avg;
	odata[index + stride] = avg;
}

/*  Averages two adjacent values and writes them back
*  - 50% coalesced
*  - Data to thread ratio 2;
*/
__global__ void D1_avg2_coal50(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

	float avg = idata[index] + idata[index + 1] / 2.;

	odata[index] = avg;
	odata[index + 1] = avg;
}

/*  Averages four adjacent values and writes them back
*  - 25% coalesced
*  - Data to thread ratio 4;
*/
__global__ void D1_avg4_coal25(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

	float avg = idata[index] + idata[index + 1] + idata[index + 2] + idata[index + 3];
	avg = avg / 4.;
	odata[index] = avg;
	odata[index + 1] = avg;
	odata[index + 2] = avg;
	odata[index + 3] = avg;
}

/*  Averages eigtht adjacent values and writes them back
*  - 12.5% coalesced
*  - Data to thread ratio 8;
*/
__global__ void D1_avg8_coal12_5(float* idata, float *odata) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

	float avg = idata[index] + idata[index + 1] + idata[index + 2] + idata[index + 3];
	avg += idata[index + 4] + idata[index + 5] + idata[index + 6] + idata[index + 7];
	avg = avg / 8.;
	odata[index] = avg;
	odata[index + 1] = avg;
	odata[index + 2] = avg;
	odata[index + 3] = avg;
	odata[index + 4] = avg;
	odata[index + 5] = avg;
	odata[index + 6] = avg;
	odata[index + 7] = avg;
}
