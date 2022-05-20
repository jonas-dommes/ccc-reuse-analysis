#ifndef FUNCTIONSTATS_H
#define FUNCTIONSTATS_H

#include "InstrStats.h" // TODO still necessary?



#define CUDA_TARGET_TRIPLE         "nvptx64-nvidia-cuda"


class FunctionStats {
public:

	// DATA
	std::map<llvm::Instruction*, InstrStats> instr_map;
	InstVector tid_calls;
	InstVector bid_calls;


	std::string function_name;
	unsigned int num_loads = 0;
	unsigned int num_stores = 0;
	unsigned int unique_loads = 0;
	unsigned int unique_stores = 0;
	unsigned int unique_total = 0;
	bool is_kernel = false;

	// METHODS
	void analyseFunction(llvm::Function &F, llvm::LoopInfo *LI);

	bool isKernel(llvm::Function &F);

	void printFunctionStats();
	void printInstrMap();

private:
};


#endif
