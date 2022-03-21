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

		struct pass_stats stats;
		stats.function_name = F.getName();

		for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

			if (isa<StoreInst>(*I)) {

				stats.num_stores++;
				store_addresses.insert(I->getOperand(1));

			} else if (isa<LoadInst>(*I)) {

				stats.num_loads++;
				load_addresses.insert(I->getOperand(0));

			}
		}

		stats.unique_loads = load_addresses.size();
		stats.unique_stores = store_addresses.size();

		// Get total unique loads and stores
		std::set<Value *> total;
		set_union(load_addresses.begin(), load_addresses.end(), store_addresses.begin(), store_addresses.end(), std::inserter(total, total.begin()));
		stats.unique_total = total.size();

		print_stats(&stats);


		// errs() << stats.function_name.c_str() << "Kernel: " << isKernel << "\n";

		return false;
	}


private:

	void print_stats(struct pass_stats *stats) {

		printf("%s\n", stats->function_name.c_str());
		printf("\tNum loads  (unique): %2d (%2d)\n", stats->num_loads, stats->unique_loads);
		printf("\tNum stores (unique): %2d (%2d)\n", stats->num_stores, stats->unique_stores);
		printf("\tNum total  (unique): %2d (%2d)\n", stats->num_stores + stats->num_loads, stats->unique_total);

	}
};




char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
