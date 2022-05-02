#ifndef FUNCTIONSTATS_H
#define FUNCTIONSTATS_H

#define CUDA_TARGET_TRIPLE         "nvptx64-nvidia-cuda"


class FunctionStats {
public:
	std::string function_name;
	unsigned int num_loads = 0;
	unsigned int num_stores = 0;
	unsigned int unique_loads = 0;
	unsigned int unique_stores = 0;
	unsigned int unique_total = 0;
	bool is_kernel = false;

	bool isKernel(llvm::Function &F);

	void printFunctionStats();
	void printInstrMap();
};


#endif
