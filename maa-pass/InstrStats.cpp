#include <iostream>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>


#include "InstrStats.h"




unsigned int InstrStats::getLoopDepth(llvm::LoopInfo *LI, llvm::Instruction *I) {

	this->loop_depth = LI->getLoopDepth((I->getParent()));

	return this->loop_depth;
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
