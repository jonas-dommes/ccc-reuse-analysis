#ifndef UTILITY_CUH
#define UTILITY_CUH

struct data_float {
	int datasize;
	float* h_idata;
	float* h_odata;
	float* d_idata;
	float* d_odata;
};

struct data_int {
	int datasize;
	int* h_idata;
	int* h_odata;
	int* d_idata;
	int* d_odata;
};

cudaError_t checkCuda(cudaError_t result);

int check_result(const float *reference, const float *result, int n);

void print_args(int argc, char **argv);

void init_data_float(struct data_float* data, int datasize);
void free_data_float(struct data_float* data);
void init_random_float(float *array, int n);

void init_data_int(struct data_int* data, int datasize);
void free_data_int(struct data_int* data);
void init_random_int(int *array, int n);


#endif
