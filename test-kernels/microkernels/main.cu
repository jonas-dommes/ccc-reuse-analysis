#include <stdio.h>

#include "calls_access_pattern.cuh"
#include "calls_coalescing.cuh"
#include "calls_cases.cuh"


int main(int argc, char **argv) {

	// // test_coalescing calls
	// printf("Runtime of call_D1_copy_coal100: %d ms\n", call_D1_copy_coal100());
	// printf("Runtime of call_D1_copy_coal50: %d ms\n", call_D1_copy_coal50());
	// printf("Runtime of call_D1_copy_coal25: %d ms\n", call_D1_copy_coal25());
	// printf("Runtime of call_D1_copy_coal12_5: %d ms\n", call_D1_copy_coal12_5());
	//
	// printf("Runtime of call_D1_avg2_coal100: %d ms\n", call_D1_avg2_coal100());
	// printf("Runtime of call_D1_avg2_coal50: %d ms\n", call_D1_avg2_coal50());
	// printf("Runtime of call_D1_avg4_coal25: %d ms\n", call_D1_avg4_coal25());
	// printf("Runtime of call_D1_avg8_coal12_5: %d ms\n", call_D1_avg8_coal12_5());
	//
	// // test_access_pattern calls
	// printf("Runtime of call_D2_ap_transpose_block: %d ms\n", call_D2_ap_transpose_block());
	// printf("Runtime of call_D2_ap_transpose: %d ms\n", call_D2_ap_transpose());
	// printf("Runtime of call_D2_ap_stepsize: %d ms\n", call_D2_ap_stepsize());

	// test_cases calls
	// printf("Runtime of call_D1_loop_consecutive: %d ms\n", call_D1_loop_consecutive());
	// printf("Runtime of call_D1_loop_strided: %d ms\n", call_D1_loop_strided());
	// printf("Runtime of call_D1_if_reduce: %d ms\n", call_D1_if_reduce());
	// printf("Runtime of call_D1_data_dependence: %d ms\n", call_D1_data_dependence());
	printf("Runtime of call_D1_blocks_work_same_data: %d ms\n", call_D1_blocks_work_same_data());
	printf("Runtime of call_D1_is_prime: %d ms\n", call_D1_is_prime());
	printf("Runtime of call_D1_gauss_sum: %d ms\n", call_D1_gauss_sum());

	return 0;

}
