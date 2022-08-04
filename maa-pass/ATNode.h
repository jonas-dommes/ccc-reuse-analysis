#ifndef ATNODE_H
#define ATNODE_H

#include "Operation.h"

#include <llvm/IR/Instructions.h>

#include <utility>

using namespace llvm;


class ATNode {
public:

	std::vector<ATNode*> children;
	Instruction* instr;
	Operation op;
	Value* value;

	// Constructor
	ATNode (Instruction* I);

// METHODS
	void insert(Instruction* I);

	void print();
	std::string to_string();
	std::string value_to_string();

};


#endif
