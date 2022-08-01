#ifndef FUNCTIONSTATS_H
#define FUNCTIONSTATS_H

#include "FunctionStats.fwd.h"
#include "InstrStats.fwd.h"


#define CUDA_TARGET_TRIPLE         "nvptx64-nvidia-cuda"

struct dependance_t {
	std::set<Instruction*> tid_calls;
	std::set<Instruction*> bid_calls;
	std::set<Instruction*> blocksize_calls;
	std::set<Instruction*> gridsize_calls;
};

class FunctionStats {

// DATA

public:
	struct dependance_t dep_calls;
	LoopInfo *LI;
	std::map<Instruction*, InstrStats> instr_map;

	std::string function_name;

	std::set<Value*> load_addresses;
	std::set<Value*> store_addresses;

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


// METHODS
public:
	FunctionStats(GridAnalysisPass *GAP, LoopInfo *LI);
	void analyseFunction(Function &F);

private:
	// Function Wide Analysis
	bool isKernel(Function &F);
	void getDimension();

	// Evaluation
	void evaluateUniques();
	void evaluateInstruction(InstrStats instr_stats);

	// Printing
	void printFunctionStats();
	void printInstrMap();
};


#endif
