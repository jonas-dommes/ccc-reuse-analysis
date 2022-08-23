#include <stdio.h>

#include "coal_calls.cuh"


int main(int argc, char **argv) {

	printf("Runtime of run_D1_copy_coal100: %d ms\n", run_D1_copy_coal100());
	printf("Runtime of run_D1_copy_coal50: %d ms\n", run_D1_copy_coal50());
	printf("Runtime of run_D1_copy_coal25: %d ms\n", run_D1_copy_coal25());
	printf("Runtime of run_D1_copy_coal12_5: %d ms\n", run_D1_copy_coal12_5());

	printf("Runtime of run_D1_avg2_coal100: %d ms\n", run_D1_avg2_coal100());
	printf("Runtime of run_D1_avg2_coal50: %d ms\n", run_D1_avg2_coal50());
	printf("Runtime of run_D1_avg4_coal25: %d ms\n", run_D1_avg4_coal25());
	printf("Runtime of run_D1_avg8_coal12_5: %d ms\n", run_D1_avg8_coal12_5());

	return 0;

}
