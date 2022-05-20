#include <iostream>
#include <algorithm>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>

#include "../llvm-rpc-passes/Common.h"
#include "../llvm-rpc-passes/GridAnalysisPass.h"

#include "InstrStats.h"

using namespace llvm;

void InstrStats::analyseInstr(Instruction *I, LoopInfo *LI, InstVector tid_calls) {

	this->getLoopDepth(I, LI);
	this->getTidDependence(I, tid_calls);

	if (isa<StoreInst>(I)) {

		this->addr = I->getOperand(1);
		this->is_store = true;

	} else if (isa<LoadInst>(I)) {

		this->addr = I->getOperand(0);
		this->is_load = true;
	}


}

void InstrStats::printInstrStats() {

	if (this->is_load) {
		printf("\t\tLoad Instr\n");
	} else if (this->is_store) {
		printf("\t\tStore Instr\n");
	}

	printf("\t\tLoop Depth: %d\n", this->loop_depth);
	printf("\t\tAddr: %p\n", this->addr);
	
}

unsigned int InstrStats::getLoopDepth(Instruction *I, LoopInfo *LI) {

	this->loop_depth = LI->getLoopDepth((I->getParent()));

	return this->loop_depth;
}

void InstrStats::getTidDependence(Instruction *I, InstVector tid_calls) {

	if (std::find(&*tid_calls.begin(), &*tid_calls.end(), I) != &*tid_calls.end()) {
		this->is_tid_dep = true;
		errs() << *I << "\n";
	}
}
