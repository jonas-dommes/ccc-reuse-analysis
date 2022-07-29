#ifndef INSTRSTATS_H
#define INSTRSTATS_H

// #include "InstrStats.fwd.h"
// #include "FunctionStats.fwd.h"

class InstrStats{

// DATA
public:
	unsigned int loop_depth = 0;   // 0 -> no loop
	bool is_conditional = false;
	bool is_load = false;
	bool is_store = false;

	bool is_tid_dep = false;
	bool is_bid_dep = false;
	bool is_blocksize_dep = false;
	bool is_gridsize_dep = false;
	bool first_use = false;            // Addr is used here for the first time
	llvm::Value * addr = NULL;
	std::string data_alias = "";
	std::string access_pattern = "";

	std::set<Instruction*> visited_phis;

// METHODS
public:
	void analyseInstr(llvm::Instruction *I, llvm::LoopInfo *LI, struct dependance_t dep_calls);
	void printInstrStats();

private:
	void getDataAlias(Instruction *I);
	unsigned int getLoopDepth(llvm::Instruction *I, llvm::LoopInfo *LI);
	void isConditional(llvm::Instruction *I);
	unsigned int getAddr();
	void analyseAccessPattern(llvm::Instruction *I, struct dependance_t dep_calls);
	void visitOperand(llvm::Instruction *I, struct dependance_t dep_calls);
	void recursiveVisitOperand(llvm::Instruction *I, unsigned int op, struct dependance_t dep_calls);
};


#endif
