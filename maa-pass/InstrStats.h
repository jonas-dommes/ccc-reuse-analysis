#ifndef INSTRSTATS_H
#define INSTRSTATS_H

class InstrStats {
public:
	bool is_loop;
	unsigned int loop_depth;
	bool is_tid_dep;
	bool is_bid_dep;

	void printInstrStats();

private:
	bool isLoop();
};


#endif
