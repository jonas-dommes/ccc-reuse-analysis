#include <stdio.h>

#include "mk_calls.cuh"


int main(int argc, char **argv) {

	printf("Runtime of run_copy1D_c100: %d ms\n", run_copy1D_coal100());

	return 0;

}
