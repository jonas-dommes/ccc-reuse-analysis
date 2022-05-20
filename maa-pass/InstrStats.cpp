#include <iostream>
#include <algorithm>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>

#include "../llvm-rpc-passes/Common.h"
#include "../llvm-rpc-passes/GridAnalysisPass.h"

#include "InstrStats.h"

using namespace llvm;

void InstrStats::analyseInstr(Instruction *I, LoopInfo *LI, std::set<Instruction*> tid_calls) {


	if (isa<StoreInst>(I)) {

		this->addr = I->getOperand(1);
		this->is_store = true;

	} else if (isa<LoadInst>(I)) {

		this->addr = I->getOperand(0);
		this->is_load = true;
	}

	this->getLoopDepth(I, LI);
	this->getTidDependence(I, tid_calls);
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

bool InstrStats::getTidDependence(Instruction *I, std::set<Instruction*> tid_calls) {

	std::set<Instruction*> workset;
	std::set<Instruction*> doneset;

	errs() << "\nStarting Analysis of\n\t" << *I << "\n";

	// Fill Addr Operand of load/store instr in worklist
	if (this->is_load) {
		workset.insert(cast<Instruction>(I->getOperand(0)));
			// errs() << *cast<Instruction>(I->getOperand(0)) << "\n";
	} else if (this->is_store) {
		workset.insert(cast<Instruction>(I->getOperand(1)));
	}

	// while there are instr in workset
	while (!workset.empty()) {
		// get and delete first instr from workset, add to doneset
		Instruction* instr = *workset.begin();
		workset.erase(instr);
		doneset.insert(instr);

		// check if instr is in tid_calls
		if (tid_calls.count(instr) > 0) {

			this->is_tid_dep = true;

			errs() << "Is in tid_calls: " << *instr << "\n";
			return true;
		}

		errs() << *instr << " has Operands\n";

		// add operands to workset if not in doneset
		for (auto &op : instr->operands()) {
			Instruction* op_instr = dyn_cast<Instruction>(op);

			// Continue if Operand is no Instruction
			if (op_instr == NULL) {
				continue;
			}

			if (doneset.count(op_instr) == 0) {

				workset.insert(op_instr);
				errs() << "\t" << *op_instr << "\n";
			}
		}
		// handle phis?

	}

	errs() << "\n-----------------Instruction was not relevant-----------------\n" << *I << "\n\n";

	return false;




	// if (std::find(&*tid_calls.begin(), &*tid_calls.end(), I) != &*tid_calls.end()) {
	// 	this->is_tid_dep = true;
	// 	errs() << *I << "\n";
	// }
}
