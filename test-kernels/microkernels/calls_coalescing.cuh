#ifndef CALLS_COALESCING_CUH
#define CALLS_COALESCING_CUH

#include "utility.cuh"

int call_D1_copy_coal100();
int call_D1_copy_coal50();
int call_D1_copy_coal25();
int call_D1_copy_coal12_5();

int call_D1_avg2_coal100();
int call_D1_avg2_coal50();
int call_D1_avg4_coal25();
int call_D1_avg8_coal12_5();

#endif
