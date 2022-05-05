#include <iostream>

#include <llvm/IR/Instructions.h>


#include "InstrStats.h"



void InstrStats::printInstrStats() {

	if (this->is_load) {
		printf("\t\tLoad Instr\n");
	} else if (this->is_store) {
		printf("\t\tStore Instr\n");
	}

	printf("\t\tLoop Depth: %d\n", this->loop_depth);
	printf("\t\tAddr: %p\n", this->addr);

}
