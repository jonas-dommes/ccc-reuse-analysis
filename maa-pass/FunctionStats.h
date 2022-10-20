#ifndef FUNCTIONSTATS_H
#define FUNCTIONSTATS_H

#include "InstrStats.fwd.h"

#include <llvm/IR/Instructions.h>
#include <llvm/Analysis/LoopInfo.h>

#include <string>
#include <set>

using namespace llvm;

#define CUDA_TARGET_TRIPLE         "nvptx64-nvidia-cuda"

class FunctionStats {

// DATA
public:
	LoopInfo* LI;
	DataLayout* DL;
	std::map<Instruction*, InstrStats> instr_map;

	std::string function_name;

	std::set<Value*> load_addresses;
	std::set<Value*> store_addresses;

	unsigned int max_tid_dim = 0;
	unsigned int max_bid_dim = 0;
	unsigned int max_block_dim = 0;
	unsigned int max_grid_dim = 0;

	unsigned int num_loads = 0;
	unsigned int num_stores = 0;
	unsigned int unique_loads = 0;
	unsigned int unique_stores = 0;
	unsigned int unique_total = 0;

	unsigned int l_num_tid = 0;
	unsigned int l_num_bid = 0;
	unsigned int l_num_bsd = 0;
	unsigned int l_num_gsd = 0;
	unsigned int s_num_tid = 0;
	unsigned int s_num_bid = 0;
	unsigned int s_num_bsd = 0;
	unsigned int s_num_gsd = 0;

	bool is_kernel = false;
	float reuse = 0.0;
	float avg_ce = 0.0;

// METHODS
public:
	FunctionStats(LoopInfo* LI, DataLayout* DL);
	void analyseFunction(Function& F);

private:
	// Function Wide Analysis
	bool isKernel(Function& F);
	void getDimension();

	// Evaluation
	void evaluateUniques();
	void evaluateInstruction(InstrStats instr_stats);
	void predictCE();
	void predictReuse();
	float factorGlobalMem();

	// Printing
	void printFunctionStats();
	void printInstrMap();
};


#endif
