#ifndef PASSSTATS_H
#define PASSSTATS_H

class PassStats {
public:
	std::string function_name;
	unsigned int num_loads = 0;
	unsigned int num_stores = 0;
	unsigned int unique_loads = 0;
	unsigned int unique_stores = 0;
	unsigned int unique_total = 0;

	void print_pass_stats();
};

#endif
