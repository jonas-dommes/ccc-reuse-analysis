#ifndef UTILITY_CUH
#define UTILITY_CUH

struct call_data {
	int datasize;
	float* h_idata;
	float* h_odata;
	float* d_idata;
	float* d_odata;
};

cudaError_t checkCuda(cudaError_t result);

int check_result(const float *reference, const float *result, int n);

void print_args(int argc, char **argv);

void init_call_data(struct call_data* data, int datasize);

void free_call_data(struct call_data* data);


void init_random(float *array, int n);

#endif
