#include <iostream>

#include "Stats.h"

void Stats::print_stats() {

		printf("%s\n", this->function_name.c_str());
		printf("\tNum loads  (unique): %2d (%2d)\n", this->num_loads, this->unique_loads);
		printf("\tNum stores (unique): %2d (%2d)\n", this->num_stores, this->unique_stores);
		printf("\tNum total  (unique): %2d (%2d)\n", this->num_stores + this->num_loads, this->unique_total);

	}
