#ifndef INSTRSTATS_H
#define INSTRSTATS_H

#include <llvm/Analysis/LoopInfo.h>


class InstrStats {

public:
	unsigned int loop_depth = 0;   // 0 -> no loop
	bool is_load = false;
	bool is_store = false;

	bool is_tid_dep = false;
	bool is_bid_dep = false;
	bool first_use = false;            // Addr is used here for the first time
	llvm::Value * addr = NULL;


	void analyseInstr(llvm::Instruction *I, llvm::LoopInfo *LI);

	void printInstrStats();

private:

	unsigned int getLoopDepth(llvm::Instruction *I, llvm::LoopInfo *LI);

	void getIdDependence();

	unsigned int getAddr();



};


#endif
