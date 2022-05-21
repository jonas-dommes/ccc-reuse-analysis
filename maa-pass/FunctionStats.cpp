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

void FunctionStats::analyseFunction(Function &F, LoopInfo* LI){

	std::set<Value *> load_addresses;
	std::set<Value *> store_addresses;
	std::map<Instruction*, InstrStats> instr_map;

	this->function_name = F.getName();

	errs() << "\nAnalysing " << this->function_name << "\n";

	this->isKernel(F);

	if (!this->is_kernel) {
		return;
	}

	for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

		// Only analyse store and load instructions, otherwise continue
		if (!isa<StoreInst>(*I) && !isa<LoadInst>(*I)) {
			continue;
		}

		InstrStats instr_stats;

		instr_stats.analyseInstr(&*I, LI, this->tid_calls);



		// if(isa<CallInst>(*I)) {
		// 	errs() << *I << "\n\t";
		// 	errs() << (I->getOperand(0)->getName()) << "\n";
		// }


		if (isa<StoreInst>(*I)) {

			this->num_stores++;
			store_addresses.insert(I->getOperand(1));


		} else if (isa<LoadInst>(*I)) {

			this->num_loads++;
			load_addresses.insert(I->getOperand(0));
		}

		instr_map[&*I] = instr_stats;

	}

	this->unique_loads = load_addresses.size();
	this->unique_stores = store_addresses.size();

	// Get total unique loads and stores TODO proper addresses
	std::set<Value *> total;
	set_union(load_addresses.begin(), load_addresses.end(), store_addresses.begin(), store_addresses.end(), std::inserter(total, total.begin()));
	this->unique_total = total.size();

	this->instr_map = instr_map;

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


void FunctionStats::printFunctionStats() {

	printf("%s", this->function_name.c_str());

	if (this->is_kernel) {
		printf(" is kernel function\n");
		printf("\tNum loads  (unique): %2d (%2d)\n", this->num_loads, this->unique_loads);
		printf("\tNum stores (unique): %2d (%2d)\n", this->num_stores, this->unique_stores);
		printf("\tNum total  (unique): %2d (%2d)\n", this->num_stores + this->num_loads, this->unique_total);
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
