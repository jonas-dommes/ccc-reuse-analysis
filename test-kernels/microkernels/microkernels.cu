#include <stdio.h>

#include "microkernels.cuh"

/*  Simple 1D copy kernel, every called thread copies one value
 *  - 100% coalesced
 *  - TID, BID dependent
 */
__global__ void copy1D_c100(float* idata, float *odata) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	odata[index] = idata[index];
}
