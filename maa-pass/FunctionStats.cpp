#include <iostream>
#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>


#include "NVPTXUtilities.h"




#include "FunctionStats.h"


bool FunctionStats::isKernel(llvm::Function &F) {

	bool isCUDA = F.getParent()->getTargetTriple() == CUDA_TARGET_TRIPLE;
	bool isKernel = isKernelFunction(F);

	if (!isCUDA || !isKernel) {
		return false;
	}

	this->is_kernel = true;
	return true;
}


void FunctionStats::printFunctionStats() {

	printf("%s", this->function_name.c_str());

	if (this->is_kernel) {
		printf(" is kernel function\n");
		printf("\tNum loads  (unique): %2d (%2d)\n", this->num_loads, this->unique_loads);
		printf("\tNum stores (unique): %2d (%2d)\n", this->num_stores, this->unique_stores);
		printf("\tNum total  (unique): %2d (%2d)\n", this->num_stores + this->num_loads, this->unique_total);
	} else {
		printf(" is NOT kernel function. No Analysis\n");

	}
}


void FunctionStats::printInstrMap() {

	for (auto& elem : instr_map) {

		std::string s_instr;
		llvm::raw_string_ostream ss(s_instr);
		ss << *(elem.first);

		std::cout << "  " << ss.str() << "\n";

		elem.second.printInstrStats();

	}


}
