#ifndef TEST_COALESCING_CUH
#define TEST_COALESCING_CUH


__global__ void D1_copy_coal100(float* idata, float *odata);
__global__ void D1_copy_coal50(float* idata, float *odata);
__global__ void D1_copy_coal25(float* idata, float *odata);
__global__ void D1_copy_coal12_5(float* idata, float *odata);

__global__ void D1_avg2_coal100(float* idata, float *odata);
__global__ void D1_avg2_coal50(float* idata, float *odata);
__global__ void D1_avg4_coal25(float* idata, float *odata);
__global__ void D1_avg8_coal12_5(float* idata, float *odata);






#endif
