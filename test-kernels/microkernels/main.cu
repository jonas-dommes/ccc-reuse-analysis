#include <stdio.h>

#include "calls_access_pattern.cuh"
#include "calls_coalescing.cuh"


int main(int argc, char **argv) {

	// test_coalescing calls
	printf("Runtime of run_D1_copy_coal100: %d ms\n", run_D1_copy_coal100());
	printf("Runtime of run_D1_copy_coal50: %d ms\n", run_D1_copy_coal50());
	printf("Runtime of run_D1_copy_coal25: %d ms\n", run_D1_copy_coal25());
	printf("Runtime of run_D1_copy_coal12_5: %d ms\n", run_D1_copy_coal12_5());

	printf("Runtime of run_D1_avg2_coal100: %d ms\n", run_D1_avg2_coal100());
	printf("Runtime of run_D1_avg2_coal50: %d ms\n", run_D1_avg2_coal50());
	printf("Runtime of run_D1_avg4_coal25: %d ms\n", run_D1_avg4_coal25());
	printf("Runtime of run_D1_avg8_coal12_5: %d ms\n", run_D1_avg8_coal12_5());

	// test_access_pattern calls
	printf("Runtime of run_D2_ap_transpose_block: %d ms\n", run_D2_ap_transpose_block());
	printf("Runtime of run_D2_ap_transpose: %d ms\n", run_D2_ap_transpose());
	printf("Runtime of run_D2_ap_stepsize: %d ms\n", run_D2_ap_stepsize());

	return 0;

}
