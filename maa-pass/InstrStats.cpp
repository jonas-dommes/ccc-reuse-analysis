#include "InstrStats.h"

#include "FunctionStats.h"
#include "ATNode.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

#include <iostream>
#include <algorithm>
#include <set>

using namespace llvm;

InstrStats :: InstrStats() {
}


void InstrStats :: analyseInstr(Instruction* I, FunctionStats* func_stats) {

	if (isa<StoreInst>(I)) {

		this->addr = I->getOperand(1);
		this->is_store = true;

	} else if (isa<LoadInst>(I)) {

		this->addr = I->getOperand(0);
		this->is_load = true;
	}

	errs() << "\n||||||||||||||||||||||| Create Access Tree for" << *I << "||||||||||||||||||||||| \n";
	this->root = new ATNode(I, this, nullptr);
	errs() << "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| \n";


	this->access_pattern = this->root->access_pattern_to_string();

	this->getDataAlias();
	this->getLoopDepth(I, func_stats->LI);
	this->isConditional(I);
	this->analyseAccessPattern(I);
}


void InstrStats :: printInstrStats() {

	if (this->is_load) {
		printf("\t\tLoad ");
	} else if (this->is_store) {
		printf("\t\tStore");
	}
	if (this->is_tid_dep) {
		printf("\tTID");
	}
	if (this->is_bid_dep) {
		printf("\tBID");
	}
	if (this->is_blocksize_dep) {
		printf("\tBSD");
	}
	if (this->is_gridsize_dep) {
		printf("\tGSD");
	}
	if (this->is_conditional) {
		printf("\tCOND");
	}
	printf("\n");

	printf("\t\tLoop Depth: %d\n", this->loop_depth);
	printf("\t\tAddr: %p\t\t Alias: %s\n", this->addr, this->data_alias.c_str());
	printf("\t\tAccess pattern: %s\n", this->access_pattern.c_str());
}


// private:
void InstrStats :: getDataAlias() {

	ATNode* cur_node = this->root;

	while (cur_node->instr_type != instr_t::NONE) {

		cur_node = cur_node->children[0];
	}

	this->data_alias = cur_node->name;
}


unsigned int InstrStats :: getLoopDepth(Instruction* I, LoopInfo* LI) {

	this->loop_depth = LI->getLoopDepth((I->getParent()));

	return this->loop_depth;
}


void InstrStats :: isConditional(Instruction* I) {

	std::string name = I->getParent()->getName();

	if (std::strncmp(name.c_str(), "if.", 3) == 0) {

		// errs() << I->getParent()->getName() << "\n";
		this->is_conditional = true;
	}

}


void InstrStats :: analyseAccessPattern(Instruction* I) {

}
