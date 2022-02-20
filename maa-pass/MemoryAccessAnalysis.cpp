#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"


using namespace llvm;

struct mma : public FunctionPass {
	static char ID;
	mma() : FunctionPass(ID) {}

	bool runOnFunction(Function &F) override {

		unsigned int num_maas = 0;

		for (auto &I : F) {
			// errs() << I << "\n";
			errs() << "I" << "\n";
		}

		// 	for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
		// errs() << *I << "\n";



		errs().write_escaped(F.getName()) << " uses " << num_maas << " memory access instructions\n";

		return false;
	}
}; // end of struct Hello

char mma::ID = 0;
static RegisterPass<mma> X("mma", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
