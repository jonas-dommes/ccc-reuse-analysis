#include "ATNode.h"

#include "InstrStats.h"

#include "llvm/Support/raw_ostream.h"

// #include <iostream>
// #include <string>



// Constructor
ATNode :: ATNode (op_t operation) : op(operation), children(nullptr, nullptr) {}

// METHODS
void ATNode :: print() {

	errs() << "(";
	if(this->children.first != nullptr) this->children.first->print();
	errs() << this->op.to_string();
	if(this->children.second != nullptr) this->children.second->print();
	errs() <<  ")";
}
