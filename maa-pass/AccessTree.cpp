#include "AccessTree.h"

#include "llvm/Support/raw_ostream.h"


// #include <iostream>
// #include <string>

// #include "InstrStats.h"
using namespace llvm;

// CONSTRUCTOR
AccessTree :: AccessTree(): root(nullptr) {}
AccessTree :: AccessTree(ATNode* rootNode): root(rootNode) {}

// METHODS
void AccessTree :: print() {

	if(this->root != nullptr) {
		this->root->print();
		errs() << "\n";

	}
}
