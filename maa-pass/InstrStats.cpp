#include "InstrStats.h"

#include "FunctionStats.h"
#include "ATNode.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include "llvm/Support/raw_ostream.h"


#include <iostream>
#include <algorithm>
#include <set>
#include <queue>
#include <cmath>
#include <limits>


using namespace llvm;

InstrStats :: InstrStats() {}


void InstrStats :: analyseInstr(Instruction* I, FunctionStats* func_stats) {


	// errs() << "\n_____ Analyse Instr: " << *I << " _____\n";

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
	this->predictCE();
	this->getLoopDepth(I, func_stats->LI);
	this->isConditional(I);
	this->analyseAccessPattern(I);
	this->predictReuse();
	// errs() << "\n________________________________________________________\n";
}


void InstrStats :: printInstrStats() {

	const std::string addr_space_str[6] = {"Generic", "Global", "Internal Use", "Shared", "Constant", "Local"};

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
	printf("\t\tAddr: %p\t Alias: %s [%s] (Size: %d Byte, Alignment: %d)\n", this->addr, this->data_alias.c_str(), (this->addr_space >= 0 && this->addr_space < 6) ? addr_space_str[this->addr_space].c_str() : "No Addr Space set", this->type_size, this->alignment);
	printf("\t\tAccess pattern: %s\n", this->access_pattern.c_str());

	// for (Offset* offset : root->offsets) {
	// 	printf("\t\t%s\n", offset->to_string_tid().c_str());
	// }
	// for (Offset* offset : root->offsets) {
	// 	printf("\t\t%s\n", offset->to_string_bid().c_str());
	// }
	printf("\t\tPredicted CE: %f\n", this->predicted_ce);
	printf("\t\tPredicted reuse: %f\n", this->reuse_factor);

	printf("\n");
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

void InstrStats :: predictCE() {

	std::vector<Offset*> offs_vec = this->root->offsets;
	std::set<int> warp_1_x;
	std::set<int> rest_x;
	float tmp_ce = 1.0;

	// Scaling factor if access is conditional
	float ce_scaling = this->is_conditional ? 0.9 : 1.0;

	// Get min Value
	int min[3] = {INT_MAX, INT_MAX, INT_MAX};
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 3; j++) {
			min[j] = std::min(min[j], offs_vec[i]->TidOffset[j]);
		}
	}

	// Add offset values to set
	for (int i = 0; i < 32; i++) {
		int normed_offset = offs_vec[i]->TidOffset[0] - min[0];
		if (normed_offset < 32) {
			warp_1_x.insert(normed_offset);
		} else {
			rest_x.insert(normed_offset);
		}
	}

	tmp_ce = warp_1_x.size() / 32.;

	this->predicted_ce = tmp_ce * ce_scaling;
}

void InstrStats :: predictReuse() {

	float addr_space;
	float load_store;
	float type_size;
	float ce_loopdepth;

	// No reuse if not in global memory
	addr_space = (this->addr_space <= 1)? 1.0 : 0;

	// Scale coalesced stores
	if (this->is_load) load_store = 1.0;
	if (this->is_store) {
		load_store = 0.5;
		if (this->predicted_ce > 0.99) load_store = 0.01;
	}

	// Scale wider/narrower types
	if (this->type_size < 4) type_size = 0.65 - (this->type_size / 10.0);
	if (this->type_size == 4) type_size = 0.3;
	if (this->type_size > 4) type_size = 0.01;

	// Scale CE
	if (0.2 < this->predicted_ce && this->predicted_ce < 0.6) {
		ce_loopdepth = 0.7;
	} else {
		ce_loopdepth = 0.2;
	}

	// Adjust by loopdepth
	if (this->loop_depth > 0) {
		ce_loopdepth *= (1 + this->loop_depth/10.);
	}

	float tmp_reuse = addr_space * load_store * type_size * ce_loopdepth;
	int num_factors = 4;
	this->reuse_factor = std::pow(tmp_reuse, 1.0/num_factors);
}


void InstrStats :: analyseAlias() {

	// Get all ElementPtr Instructions
	std::set<ATNode*> GEPs = this->getNodesByInstr_t(instr_t::GEP);

	// Only one is expected to be found without data dependence
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
