#ifndef INSTRSTATS_H
#define INSTRSTATS_H

// #include "InstrStats.fwd.h"
#include "FunctionStats.fwd.h"

#include <string>
#include <set>

#include <llvm/IR/Instructions.h>
#include <llvm/Analysis/LoopInfo.h>

// #include <llvm/IR/Value.h>


using namespace llvm;


#define OP0 0
#define OP1 1


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
	Value * addr = NULL;
	std::string data_alias = "";
	std::string access_pattern = "";

private:
	std::set<Instruction*> visited_phis;

// METHODS
public:
	void analyseInstr(Instruction *I, FunctionStats *func_stats);
	void printInstrStats();

private:
	void getDataAlias(Instruction *I);
	unsigned int getLoopDepth(Instruction *I, LoopInfo *LI);
	void isConditional(Instruction *I);
	unsigned int getAddr();
	void analyseAccessPattern(Instruction *I, struct dependance_t dep_calls);
	void visitOperand(Instruction *I, struct dependance_t dep_calls);
	void recursiveVisitOperand(Instruction *I, unsigned int op, struct dependance_t dep_calls);
};


#endif
