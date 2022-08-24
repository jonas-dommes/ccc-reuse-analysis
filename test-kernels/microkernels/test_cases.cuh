#ifndef TEST_CASES_CUH
#define TEST_CASES_CUH


__global__ void D1_loop_consecutive(float* idata, float* odata);
__global__ void D1_loop_strided(float* idata, float* odata);
__global__ void D1_if_reduce(float* idata, float* odata);
__global__ void D1_data_dependence(int* idata, int* odata);
__global__ void D1_blocks_work_same_data(float* idata, float *odata);
__global__ void D1_is_prime(int* idata, int *odata);
__global__ void D1_gauss_sum(int* idata, int *odata);





#endif
