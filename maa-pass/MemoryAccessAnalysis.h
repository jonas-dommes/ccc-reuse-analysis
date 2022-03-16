#ifndef MEMORYACCESSANALYSIS_H
#define MEMORYACCESSANALYSIS_H


#define CUDA_TARGET_TRIPLE         "nvptx64-nvidia-cuda"

struct pass_stats {
	std::string function_name;
	unsigned int num_loads = 0;
	unsigned int num_stores = 0;
	unsigned int unique_loads = 0;
	unsigned int unique_stores = 0;
	unsigned int unique_total = 0;
};

#endif
