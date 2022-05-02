#ifndef INSTRSTATS_H
#define INSTRSTATS_H

class InstrStats {
public:
	// bool is_loop;
	unsigned int loop_depth = 0;   // 0 -> no loop
	bool is_tid_dep = false;
	bool is_bid_dep = false;
	bool first_use = false;            // Addr is used here for the first time
	llvm::Value * addr = NULL;
	bool is_load = false;
	bool is_store = false;

	void printInstrStats();

private:
	unsigned int getLoopDepth();
	void getIdDependence();
	unsigned int getAddr();
};


#endif
