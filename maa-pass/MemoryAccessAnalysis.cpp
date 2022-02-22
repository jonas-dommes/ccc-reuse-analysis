#include <string>

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"

#include "MemoryAccessAnalysis.h"

using namespace llvm;




struct maa : public FunctionPass {
	static char ID;
	maa() : FunctionPass(ID) {}

	bool runOnFunction(Function &F) override {

		struct pass_stats stats;
		stats.function_name = F.getName();

		for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

			if (isa<StoreInst>(*I)) {

				stats.num_stores++;

			} else if (isa<LoadInst>(*I)) {

				stats.num_loads++;
			}
		}

		print_stats(&stats);

		return false;
	}


private:

	void print_stats(struct pass_stats *stats) {

		printf("%s\n", stats->function_name.c_str());
		printf("\tNum loads:\t%4d\n", stats->num_loads);
		printf("\tNum stores:\t%4d\n", stats->num_stores);

	}
};




char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
