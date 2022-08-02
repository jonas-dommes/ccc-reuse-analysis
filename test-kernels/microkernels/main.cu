#include <stdio.h>

#include "mk_calls.cuh"


int main(int argc, char **argv) {

	int ms = run_copy1D_c100();

	printf("Runtime of run_copy1D_c100: %d ms\n", ms);

	return 0;

}
