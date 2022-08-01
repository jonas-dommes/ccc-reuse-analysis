#ifndef MICROKERNELS_CUH
#define MICROKERNELS_CUH

__global__ void TEST_access_pattern(float *odata);

__global__ void TEST_nested_conditions(float *odata);

__global__ void TEST_for_loops(float *odata, const float *idata);

__global__ void TEST_nested_for_loops(float *odata, const float *idata);

__global__ void kernel_a(float *odata, const float *idata, int work_per_thread);

__global__ void kernel_b(float *odata, const float *idata, int work_per_thread);

__global__ void reduce1(int *d_data);

__global__ void reduce3(int *d_data);

__global__ void copy(float *odata, const float *idata);

__global__ void transposeSimple(float *odata, const float *idata);


#endif
