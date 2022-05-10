#ifndef INSTRSTATS_H
#define INSTRSTATS_H

#include <llvm/Analysis/LoopInfo.h>

class InstrStats {
public:
	// bool is_loop;
	unsigned int loop_depth = 0;   // 0 -> no loop
	bool is_load = false;
	bool is_store = false;

	bool is_tid_dep = false;
	bool is_bid_dep = false;
	bool first_use = false;            // Addr is used here for the first time
	llvm::Value * addr = NULL;

	void printInstrStats();
	unsigned int getLoopDepth(llvm::LoopInfo *LI, llvm::Instruction *I);
	void getIdDependence();
	unsigned int getAddr();

};


#endif
