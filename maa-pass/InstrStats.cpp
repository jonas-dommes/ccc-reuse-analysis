#include "InstrStats.h"

#include "FunctionStats.h"
#include "ATNode.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

#include <iostream>
#include <algorithm>
#include <set>
#include <queue>


using namespace llvm;

InstrStats :: InstrStats() {}


void InstrStats :: analyseInstr(Instruction* I, FunctionStats* func_stats) {


	errs() << "\n_____ Analyse Instr: " << *I << " _____\n";

	if (StoreInst* storeInst = dyn_cast<StoreInst>(I)) {

		this->addr = storeInst->getPointerOperand();
		this->alignment = storeInst->getAlignment();
		this->is_store = true;

	} else if (LoadInst* loadInst = dyn_cast<LoadInst>(I)) {

		this->addr = loadInst->getPointerOperand();
		this->alignment = loadInst->getAlignment();
		this->is_load = true;
	}

	this->setTypeSize(I, func_stats);

	this->root = new ATNode(I, this, nullptr);

	this->access_pattern = this->root->access_pattern_to_string();

	this->analyseAlias();
	this->analyseOffset();
	this->getLoopDepth(I, func_stats->LI);
	this->isConditional(I);
	this->analyseAccessPattern(I);
	errs() << "\n________________________________________________________\n";
}


void InstrStats :: printInstrStats() {

	if (this->is_load) {
		printf("\t\tLoad ");
	} else if (this->is_store) {
		printf("\t\tStore");
	}
	if (this->tid_dim > 0) {
		printf("\tTID(%d)", this->tid_dim);
	}
	if (this->bid_dim > 0) {
		printf("\tBID(%d)", this->bid_dim);
	}
	if (this->block_dim > 0) {
		printf("\tBDim(%d)", this->block_dim);
	}
	if (this->grid_dim > 0) {
		printf("\tGDim(%d)", this->grid_dim);
	}
	if (this->is_conditional) {
		printf("\tCOND");
	}
	printf("\n");

	printf("\t\tLoop Depth: %d\n", this->loop_depth);
	printf("\t\tAddr: %p\t\t Alias: %s(%d Byte, Space %d)\n", this->addr, this->data_alias.c_str(), this->type_size, this->addr_space);
	printf("\t\tAddr: %p\t\t Alias: %s(Size: %d Byte, Space: %d, Alignment: %d)\n", this->addr, this->data_alias.c_str(), this->type_size, this->addr_space, this->alignment);
	printf("\t\tAccess pattern: %s\n", this->access_pattern.c_str());
	printf("\t\tTID Offset: %d, %d, %d\n", root->offset.TidOffset[0], root->offset.TidOffset[1], root->offset.TidOffset[2]);
	printf("\t\tBID Offset: %d, %d, %d\n", root->offset.BidOffset[0], root->offset.BidOffset[1], root->offset.BidOffset[2]);
}


// private:
void InstrStats :: analyseOffset() {

	if (this->type_size == 0) {
		errs() << "[analyseOffset()] Type size is not set. Returning.\n";
		return;
	}

	this->root->calcOffset();
	this->root->offsetMulDep();
}


void InstrStats :: analyseAlias() {

	// Get all ElementPtr Instructions
	std::set<ATNode*> GEPs = this->getNodesByInstr_t(instr_t::GEP);

	// Only one is expected to be found
	if (GEPs.size() != 1) {
		errs() << "[getDataAlias()] Found unexpected number of GEPs: " << GEPs.size() << "\n";
		return;
	}

	ATNode* GEP_node = *GEPs.begin();
	this->data_alias = GEP_node->children[0]->name;

	GetElementPtrInst* GEP_instr = cast<GetElementPtrInst>(GEP_node->value);
	this->addr_space = GEP_instr->getAddressSpace();
}


void InstrStats :: setTypeSize(Instruction* I, FunctionStats* func_stats) {

	if (StoreInst* storeInst = dyn_cast<StoreInst>(I)) {

		this->type_size = func_stats->DL->getTypeAllocSize(storeInst->getPointerOperandType()->getPointerElementType());

	} else if (LoadInst* loadInst = dyn_cast<LoadInst>(I)) {

		this->type_size = func_stats->DL->getTypeAllocSize(loadInst->getPointerOperandType()->getPointerElementType());
	} else {
		errs() << "Called setTypeSize on invalid Instruction\n";
	}
}

std::set<ATNode*> InstrStats :: getNodesByInstr_t(instr_t instr_type) {

	std::queue<ATNode*> worklist;
	std::set<ATNode*> result;

	worklist.push(this->root);

	while (!worklist.empty()) {

		ATNode* cur_node = worklist.front();
		worklist.pop();

		if (cur_node->instr_type == instr_type) result.insert(cur_node);

		for (ATNode* child : cur_node->children) {
			worklist.push(child);
		}
	}
	return result;
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
