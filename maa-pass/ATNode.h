#ifndef ATNODE_H
#define ATNODE_H

#include "Operation.h"
#include <utility>


class ATNode {
public:

	std::pair<ATNode*, ATNode*> children;
	Operation op;

	// Constructor
	ATNode (op_t operation);

// METHODS
	void print();

};


#endif
