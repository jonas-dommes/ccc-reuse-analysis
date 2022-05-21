#ifndef FUNCTIONSTATS_FWD_H
#define FUNCTIONSTATS_FWD_H


class FunctionStats;

// struct dependance_t;

// class FunctionStats {
// public:
//
// 	// DATA
// 	std::map<llvm::Instruction*, InstrStats> instr_map;
//
// 	struct dependance_t {
// 		std::set<Instruction*> tid_calls;
// 		std::set<Instruction*> bid_calls;
// 		std::set<Instruction*> blocksize_calls;
// 		std::set<Instruction*> gridsize_calls;
// 	} dep_calls;
//
//
// 	std::string function_name;
// 	unsigned int num_loads = 0;
// 	unsigned int num_stores = 0;
// 	unsigned int unique_loads = 0;
// 	unsigned int unique_stores = 0;
// 	unsigned int unique_total = 0;
// 	bool is_kernel = false;
//
// 	// METHODS
// 	void analyseFunction(llvm::Function &F, llvm::LoopInfo *LI);
//
// 	bool isKernel(llvm::Function &F);
//
// 	void printFunctionStats();
// 	void printInstrMap();
//
// private:
// };


#endif
