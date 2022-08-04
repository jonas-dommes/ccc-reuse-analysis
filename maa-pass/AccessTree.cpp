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
		printf("\n");

	}
}

std::string AccessTree :: to_string() {

	std::string tree_string;

	if(this->root != nullptr) {
		tree_string.append(this->root->to_string());
	}
	tree_string.append("\n");

	return tree_string;
}
