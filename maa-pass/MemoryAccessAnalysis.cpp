#include <string>
#include <set>
#include <iostream>
#include <map>

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Analysis/LoopInfo.h>

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

	void getAnalysisUsage(AnalysisUsage &AU) const override {
		AU.setPreservesCFG();
		AU.addRequired<LoopInfoWrapperPass>();
	}


	bool runOnFunction(Function &F) override {

		FunctionStats func_stats;

		LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

		func_stats.analyseFunction(F, &LI);

		return false;
	}


private:

};




char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
