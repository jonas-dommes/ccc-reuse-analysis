#ifndef INSTRSTATS_H
#define INSTRSTATS_H

class InstrStats {
public:
	bool isLoop;
	unsigned int loopDepth;
	bool isTidDep;
	bool isBidDep;

	void printInstrStats();
};


#endif
