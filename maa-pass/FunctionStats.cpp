#include "InstrStats.h"

#include "FunctionStats.h"

#include "NVPTXUtilities.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <string>
#include <set>
#include <cmath>


using namespace llvm;



FunctionStats :: FunctionStats(LoopInfo* LI, DataLayout* DL) : LI(LI), DL(DL) {}


void FunctionStats :: analyseFunction(Function& F){

	this->function_name = F.getName();

	this->isKernel(F);

	if (!this->is_kernel) {
		return;
	}

	errs() << "\n###################### Analysing " << this->function_name << " ######################\n\n";


	for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

		// Only analyse store and load instructions, otherwise continue
		if (!isa<StoreInst>(*I) && !isa<LoadInst>(*I)) {
			continue;
		}

		InstrStats instr_stats;

		instr_stats.analyseInstr(&*I, this);
		this->evaluateInstruction(instr_stats);

		this->instr_map.emplace(&*I, instr_stats);
	}

	this->getDimension();
	this->evaluateUniques();
	this->predictReuse();
	this->predictCE();

	// Print results
	this->printFunctionStats();
	this->printInstrMap();
}


bool FunctionStats :: isKernel(Function& F) {

	bool isCUDA = F.getParent()->getTargetTriple() == CUDA_TARGET_TRIPLE;
	bool isKernel = isKernelFunction(F);

	if (!isCUDA || !isKernel) {
		return false;
	}

	this->is_kernel = true;
	return true;
}


void FunctionStats :: getDimension() {

	unsigned int tid_dim = 0;
	unsigned int bid_dim = 0;
	unsigned int block_dim = 0;
	unsigned int grid_dim = 0;

	for (const auto& [instr, stats]: this->instr_map) {

		if (stats.tid_dim > tid_dim) tid_dim = stats.tid_dim;
		if (stats.bid_dim > bid_dim) bid_dim = stats.bid_dim;
		if (stats.block_dim > block_dim) block_dim = stats.block_dim;
		if (stats.grid_dim > grid_dim) grid_dim = stats.grid_dim;
	}

	this->max_tid_dim = tid_dim;
	this->max_bid_dim = bid_dim;
	this->max_block_dim = block_dim;
	this->max_grid_dim = grid_dim;
}


void FunctionStats :: evaluateUniques() {

	this->unique_loads = this->load_addresses.size();
	this->unique_stores = this->store_addresses.size();

	// Get total unique loads and stores TODO proper addresses
	std::set<Value*> total;
	set_union(load_addresses.begin(), load_addresses.end(), store_addresses.begin(), store_addresses.end(), std::inserter(total, total.begin()));
	this->unique_total = total.size();
}


void FunctionStats :: evaluateInstruction(InstrStats instr_stats) {

	if (instr_stats.is_load) {

		this->num_loads++;
		this->load_addresses.insert(instr_stats.addr);

		if (instr_stats.tid_dim > 0) {
			this->l_num_tid++;
		}
		if (instr_stats.tid_dim > 0) {
			this->l_num_bid++;
		}
		if (instr_stats.block_dim > 0) {
			this->l_num_bsd++;
		}
		if (instr_stats.grid_dim > 0) {
			this->l_num_gsd++;
		}
	} else if (instr_stats.is_store) {

		this->num_stores++;
		this->store_addresses.insert(instr_stats.addr);

		if (instr_stats.tid_dim > 0) {
			this->s_num_tid++;
		}
		if (instr_stats.tid_dim > 0) {
			this->s_num_bid++;
		}
		if (instr_stats.block_dim > 0) {
			this->s_num_bsd++;
		}
		if (instr_stats.grid_dim > 0) {
			this->s_num_gsd++;
		}
	}
}
void FunctionStats :: predictCE() {

	float total_weight = 0.0;
	float tmp_ce = 0.0;

	for (const auto& [instr, stats]: this->instr_map) {

		if (stats.addr_space <= 1) {

			float tmp_weight = 1.0;
			if (stats.loop_depth > 0) {
				tmp_weight += 0.3 *  stats.loop_depth;
			}

			if (stats.is_conditional) {
				tmp_weight -= 0.2;
			}

			tmp_ce += stats.predicted_ce * (1.0 + tmp_weight/2.0);
			total_weight += 1.0 + tmp_weight;

		}
	}
	this->avg_ce = tmp_ce / total_weight;
	errs() << "Debug vals:\n";
	errs() << "tmp_ce: " << tmp_ce << "\n";
	errs() << "total_weight: " << total_weight << "\n";

	errs() << "Avg Ce: " << this->avg_ce << "\n";
}

void FunctionStats :: predictReuse() {

	int num_instr = 0;
	float tmp_reuse = 1.0;

	for (const auto& [instr, stats]: this->instr_map) {

		if (stats.addr_space <= 1) {
			num_instr++;
			tmp_reuse *= stats.reuse_factor;
		}
	}
	tmp_reuse = pow (tmp_reuse, 1./num_instr);

	float tmp_unique = this->unique_total / (float) (this->num_loads + this->num_stores);
	tmp_unique = 1 - 0.5 * tmp_unique;

	errs() << "Debug vals:\n";
	errs() << "tmp_reuse: " << tmp_reuse << "\n";
	errs() << "tmp_unique: " << tmp_unique << "\n";
	errs() << "this->factorGlobalMem(): " << this->factorGlobalMem() << "\n";

	this->reuse = 0.7 * tmp_reuse + 0.1 * tmp_unique + 0.2 * this->factorGlobalMem();

	errs() << "Reuse:  " << this->reuse << "\n";
}


float FunctionStats :: factorGlobalMem() {

	int num_globals = 0;
	for (const auto& [instr, stats]: this->instr_map) {
		if (stats.addr_space <= 1) {
			num_globals++;
		}
	}
	float ratio = num_globals / (float) (this->num_loads + this->num_stores);
	return ratio * ratio;
}

void FunctionStats :: printFunctionStats() {

	printf("\n%s", this->function_name.c_str());

	if (this->is_kernel) {
		printf(" is kernel function. BlockDim(%d/%d), GridDim(%d/%d)\n", this->max_tid_dim, this->max_block_dim, this->max_bid_dim, this->max_grid_dim);
		// printf("\tNum loads  (unique): %2d (%2d)\n", this->num_loads, this->unique_loads);
		// printf("\tNum stores (unique): %2d (%2d)\n", this->num_stores, this->unique_stores);
		// printf("\tNum total  (unique): %2d (%2d)\n", this->num_stores + this->num_loads, this->unique_total);

		printf("\t%6s | %4s | %4s | %4s | %4s | %4s | %4s \n", "", "num", "uni", "tid", "bid", "bsd", "gsd");
		printf("\t-------------------------------------------------\n");
		printf("\t%6s | %4d | %4d | %4d | %4d | %4d | %4d \n", "loads ", this->num_loads, this->unique_loads, this->l_num_tid, this->l_num_bid, this->l_num_bsd, this->l_num_gsd);
		printf("\t%6s | %4d | %4d | %4d | %4d | %4d | %4d \n", "stores", this->num_stores, this->unique_stores, this->s_num_tid, this->s_num_bid, this->s_num_bsd, this->s_num_gsd);
		printf("\t%6s | %4d | %4d | %4d | %4d | %4d | %4d \n", "total ", this->num_loads + this->num_stores, this->unique_total, this->l_num_tid + this->s_num_tid, this->l_num_bid + this->s_num_bid, this->l_num_bsd + this->s_num_bsd, this->l_num_gsd + this->s_num_gsd);
		printf("\tAvg CE: %f\n", this->avg_ce);
		printf("\tPredicted Reuse: %f\n", this->reuse);


		printf("\n");



	} else {
		printf(" is NOT kernel function. No Analysis\n");
	}
}


void FunctionStats :: printInstrMap() {

	for (auto& elem : instr_map) {

		std::string s_instr;
		raw_string_ostream ss(s_instr);
		ss << *(elem.first);

		std::cout << "  " << ss.str() << "\n";

		elem.second.printInstrStats();
	}
}
