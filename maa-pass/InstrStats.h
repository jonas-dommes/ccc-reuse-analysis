#ifndef INSTRSTATS_H
#define INSTRSTATS_H

class InstrStats {
public:
	// bool is_loop;
	unsigned int loop_depth;   // 0 -> no loop
	bool is_tid_dep;
	bool is_bid_dep;
	bool first_use;            // Addr is used here for the first time
	unsigned int addr;         // TODO change type to address type
	bool is_load;
	bool is_store;

	void printInstrStats();

private:
	unsigned int getLoopDepth();
	void getIdDependence();
	unsigned int getAddr();
};


#endif
