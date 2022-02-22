#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"


using namespace llvm;

struct maa : public FunctionPass {
	static char ID;
	maa() : FunctionPass(ID) {}

	bool runOnFunction(Function &F) override {

		unsigned int num_loads = 0;
		unsigned int num_stores = 0;

	for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

		if (isa<StoreInst>(*I)) {

			num_stores++;



		} else if (isa<LoadInst>(*I)) {

			num_loads++;
		}
	}


	errs().write_escaped(F.getName()) << " uses " << num_stores + num_loads << " memory access instructions\n";

	return false;
}
}; // end of struct Hello

char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
