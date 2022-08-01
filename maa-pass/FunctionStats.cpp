#include <iostream>
#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"

#include <llvm/Support/raw_ostream.h>

#include "../llvm-rpc-passes/Common.h"
#include "../llvm-rpc-passes/GridAnalysisPass.h"
#include "NVPTXUtilities.h"

#include "InstrStats.h"

#include "FunctionStats.h"


using namespace llvm;

FunctionStats::FunctionStats(GridAnalysisPass *GAP, LoopInfo *LI) {

	this->LI = LI;

	// Copy Tid dependent Instructions
	for (auto& call : GAP->getThreadIDDependentInstructions()) {
		this->dep_calls.tid_calls.insert(call);
	}

	// Copy Bid dependent Instructions
	for (auto& call : GAP->getBlockIDDependentInstructions()) {
		this->dep_calls.bid_calls.insert(call);
	}

	// Copy Blocksize dependent Instructions
	for (auto& call : GAP->getBlockSizeDependentInstructions()) {
		this->dep_calls.blocksize_calls.insert(call);
	}

	// Copy Gridsize dependent Instructions
	for (auto& call : GAP->getGridSizeDependentInstructions()) {
		this->dep_calls.gridsize_calls.insert(call);
	}

	// errs() << "Found " << this->dep_calls.tid_calls.size() << " TID_calls\n";
	// for (auto& call : this->dep_calls.tid_calls) {
	// 	errs() << *call << "\n";
	// }
	// errs() << "Found " << this->dep_calls.bid_calls.size() << " BID_calls\n";
	// for (auto& call : this->dep_calls.bid_calls) {
	// 	errs() << *call << "\n";
	// }
	// errs() << "Found " << this->dep_calls.blocksize_calls.size() << " blocksize_calls\n";
	// for (auto& call : this->dep_calls.blocksize_calls) {
	// 	errs() << *call << "\n";
	// }
	// errs() << "Found " << this->dep_calls.gridsize_calls.size() << " gridsize_calls\n";
	// for (auto& call : this->dep_calls.gridsize_calls) {
	// 	errs() << *call << "\n";
	// }
}


void FunctionStats::analyseFunction(Function &F){

	this->function_name = F.getName();

	this->isKernel(F);

	if (!this->is_kernel) {
		return;
	}

	errs() << "\n###################### Analysing " << this->function_name << " ######################\n\n";

	this->getDimension();

	for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

		// Only analyse store and load instructions, otherwise continue
		if (!isa<StoreInst>(*I) && !isa<LoadInst>(*I)) {
			continue;
		}

		InstrStats instr_stats;

		instr_stats.analyseInstr(&*I, this);
		this->evaluateInstruction(instr_stats);

		instr_map[&*I] = instr_stats;
	}

	this->evaluateUniques();


	this->printFunctionStats();
	this->printInstrMap();
}


bool FunctionStats::isKernel(Function &F) {

	bool isCUDA = F.getParent()->getTargetTriple() == CUDA_TARGET_TRIPLE;
	bool isKernel = isKernelFunction(F);

	if (!isCUDA || !isKernel) {
		return false;
	}

	this->is_kernel = true;
	return true;
}


void FunctionStats::getDimension() {

	unsigned int block_dim = 0;
	unsigned int grid_dim = 0;

	std::map<char, unsigned int> char_map {{'x', 1}, {'y', 2}, {'z', 3}};

	for (auto& call : this->dep_calls.tid_calls) {

		StringRef name = cast<CallInst>(call)->getCalledFunction()->getName();
		if (char_map[name.back()] > block_dim) block_dim = char_map[name.back()];
	}

	for (auto& call : this->dep_calls.bid_calls) {

		StringRef name = cast<CallInst>(call)->getCalledFunction()->getName();
		if (char_map[name.back()] > grid_dim) grid_dim = char_map[name.back()];
	}

	for (auto& call : this->dep_calls.blocksize_calls) {

		StringRef name = cast<CallInst>(call)->getCalledFunction()->getName();
		if (char_map[name.back()] > block_dim) block_dim = char_map[name.back()];
	}

	for (auto& call : this->dep_calls.gridsize_calls) {

		StringRef name = cast<CallInst>(call)->getCalledFunction()->getName();
		if (char_map[name.back()] > grid_dim) grid_dim = char_map[name.back()];
	}

	this->max_block_dim = block_dim;
	this->max_grid_dim = grid_dim;
}


void FunctionStats::evaluateUniques() {

	this->unique_loads = this->load_addresses.size();
	this->unique_stores = this->store_addresses.size();

	// Get total unique loads and stores TODO proper addresses
	std::set<Value *> total;
	set_union(load_addresses.begin(), load_addresses.end(), store_addresses.begin(), store_addresses.end(), std::inserter(total, total.begin()));
	this->unique_total = total.size();
}


void FunctionStats::evaluateInstruction(InstrStats instr_stats) {

	if (instr_stats.is_load) {

		this->num_loads++;
		this->load_addresses.insert(instr_stats.addr);

		if (instr_stats.is_tid_dep) {
			this->l_num_tid++;
		}
		if (instr_stats.is_bid_dep) {
			this->l_num_bid++;
		}
		if (instr_stats.is_blocksize_dep) {
			this->l_num_bsd++;
		}
		if (instr_stats.is_gridsize_dep) {
			this->l_num_gsd++;
		}
	} else if (instr_stats.is_store) {

		this->num_stores++;
		this->store_addresses.insert(instr_stats.addr);

		if (instr_stats.is_tid_dep) {
			this->s_num_tid++;
		}
		if (instr_stats.is_bid_dep) {
			this->s_num_bid++;
		}
		if (instr_stats.is_blocksize_dep) {
			this->s_num_bsd++;
		}
		if (instr_stats.is_gridsize_dep) {
			this->s_num_gsd++;
		}
	}
}


void FunctionStats::printFunctionStats() {

	printf("\n%s", this->function_name.c_str());

	if (this->is_kernel) {
		printf(" is kernel function. BlockDim(%d), GridDim(%d)\n", this->max_block_dim, this->max_grid_dim);
		// printf("\tNum loads  (unique): %2d (%2d)\n", this->num_loads, this->unique_loads);
		// printf("\tNum stores (unique): %2d (%2d)\n", this->num_stores, this->unique_stores);
		// printf("\tNum total  (unique): %2d (%2d)\n", this->num_stores + this->num_loads, this->unique_total);

		printf("\t%6s | %4s | %4s | %4s | %4s | %4s | %4s \n", "", "num", "uni", "tid", "bid", "bsd", "gsd");
		printf("\t-------------------------------------------------\n");
		printf("\t%6s | %4d | %4d | %4d | %4d | %4d | %4d \n", "loads ", this->num_loads, this->unique_loads, this->l_num_tid, this->l_num_bid, this->l_num_bsd, this->l_num_gsd);
		printf("\t%6s | %4d | %4d | %4d | %4d | %4d | %4d \n", "stores", this->num_stores, this->unique_stores, this->s_num_tid, this->s_num_bid, this->s_num_bsd, this->s_num_gsd);
		printf("\t%6s | %4d | %4d | %4d | %4d | %4d | %4d \n\n", "total ", this->num_loads + this->num_stores, this->unique_total, this->l_num_tid + this->s_num_tid, this->l_num_bid + this->s_num_bid, this->l_num_bsd + this->s_num_bsd, this->l_num_gsd + this->s_num_gsd);



	} else {
		printf(" is NOT kernel function. No Analysis\n");
	}
}


void FunctionStats::printInstrMap() {

	for (auto& elem : instr_map) {

		std::string s_instr;
		raw_string_ostream ss(s_instr);
		ss << *(elem.first);

		std::cout << "  " << ss.str() << "\n";

		elem.second.printInstrStats();
	}
}
