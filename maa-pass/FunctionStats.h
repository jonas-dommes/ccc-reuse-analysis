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
protected:
	struct dependance_t dep_calls;

private:
	std::map<llvm::Instruction*, InstrStats> instr_map;

	std::string function_name;

	std::set<Value*> load_addresses;
	std::set<Value*> store_addresses;

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
	FunctionStats(GridAnalysisPass *GAP);
	void analyseFunction(llvm::Function &F, llvm::LoopInfo *LI);

private:
	bool isKernel(llvm::Function &F);
	void evaluateInstruction(InstrStats instr_stats, std::set<Value *> *load_addresses, std::set<Value *> *store_addresses);
	void evaluateUniques(std::set<Value *> load_addresses, std::set<Value *> store_addresses);
	void printFunctionStats();
	void printInstrMap();
};


#endif
