#ifndef INSTRSTATS_H
#define INSTRSTATS_H

// #include "InstrStats.fwd.h"
#include "FunctionStats.fwd.h"
#include "ATNode.fwd.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Analysis/LoopInfo.h>

#include <string>
#include <set>

using namespace llvm;

class InstrStats{

// DATA
public:
	ATNode* root;

	unsigned int loop_depth = 0;   // 0 -> no loop
	bool is_conditional = false;
	bool is_load = false;
	bool is_store = false;

	bool is_tid_dep = false;
	bool is_bid_dep = false;
	bool is_blocksize_dep = false;
	bool is_gridsize_dep = false;
	bool first_use = false;            // Addr is used here for the first time
	Value* addr = nullptr;
	std::string data_alias = "";
	std::string access_pattern = "";

private:

public:
	// CONSTRUCTOR
	InstrStats();

// METHODS
public:
	void analyseInstr(Instruction* I, FunctionStats* func_stats);
	void printInstrStats();

private:
	void getDataAlias();
	unsigned int getLoopDepth(Instruction* I, LoopInfo* LI);
	void isConditional(Instruction* I);
	unsigned int getAddr();
	void analyseAccessPattern(Instruction* I);

};


#endif
