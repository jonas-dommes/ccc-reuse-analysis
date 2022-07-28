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

#include "../llvm-rpc-passes/Common.h"
#include "../llvm-rpc-passes/GridAnalysisPass.h"

#include "NVPTXUtilities.h"

#include "PassStats.h"
#include "FunctionStats.h"
#include "InstrStats.h"
#include "Util.h"

#include "MemoryAccessAnalysis.h"

using namespace llvm;


struct maa : public FunctionPass {

	static char ID;
	maa() : FunctionPass(ID) {}

	void getAnalysisUsage(AnalysisUsage &AU) const override {
		// AU.setPreservesCFG();
		AU.addRequired<LoopInfoWrapperPass>();
		AU.addRequired<GridAnalysisPass>();
	}


	bool runOnFunction(Function &F) override {

		FunctionStats func_stats;

		// getAnalysis<LoopSimplifyID>(F);
		LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
		GridAnalysisPass *GAP = &getAnalysis<GridAnalysisPass>();

		// Copy Tid dependent Instructions
		for (auto& call : GAP->getThreadIDDependentInstructions()) {
			func_stats.dep_calls.tid_calls.insert(call);
		}

		// Copy Bid dependent Instructions
		for (auto& call : GAP->getBlockIDDependentInstructions()) {
			func_stats.dep_calls.bid_calls.insert(call);
		}

		// Copy Blocksize dependent Instructions
		for (auto& call : GAP->getBlockSizeDependentInstructions()) {
			func_stats.dep_calls.blocksize_calls.insert(call);
		}

		// Copy Gridsize dependent Instructions
		for (auto& call : GAP->getGridSizeDependentInstructions()) {
			func_stats.dep_calls.gridsize_calls.insert(call);
		}

		// for (auto const &call : func_stats.dep_calls.tid_calls) {
		// 	errs() << *call << "\n";
		// }

		func_stats.analyseFunction(F, &LI);

		return false;
	}


private:

};




char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
