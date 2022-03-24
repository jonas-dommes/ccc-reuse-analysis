#include <string>
#include <set>
#include <iostream>

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"

#include "NVPTXUtilities.h"

#include "MemoryAccessAnalysis.h"
#include "PassStats.h"
#include "FunctionStats.h"
#include "InstrStats.h"
#include "Util.h"

using namespace llvm;


struct maa : public FunctionPass {

	static char ID;
	maa() : FunctionPass(ID) {}


	bool runOnFunction(Function &F) override {

		std::set<Value *> load_addresses;
		std::set<Value *> store_addresses;

		// Stop if function is no kernelfunction
		bool isCUDA = F.getParent()->getTargetTriple() == CUDA_TARGET_TRIPLE;
		bool isKernel = isKernelFunction(F);

		if (!isCUDA || !isKernel) {
			return false;
		}

		FunctionStats func_stats;
		func_stats.function_name = F.getName();

		for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

			if (isa<StoreInst>(*I)) {

				func_stats.num_stores++;
				store_addresses.insert(I->getOperand(1));

			} else if (isa<LoadInst>(*I)) {

				func_stats.num_loads++;
				load_addresses.insert(I->getOperand(0));

			}
		}

		func_stats.unique_loads = load_addresses.size();
		func_stats.unique_stores = store_addresses.size();

		// Get total unique loads and stores
		std::set<Value *> total;
		set_union(load_addresses.begin(), load_addresses.end(), store_addresses.begin(), store_addresses.end(), std::inserter(total, total.begin()));
		func_stats.unique_total = total.size();

		// Util::print_stats(&stats);
		func_stats.printFunctionStats();


		// errs() << stats.function_name.c_str() << "Kernel: " << isKernel << "\n";

		return false;
	}


private:

};




char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
