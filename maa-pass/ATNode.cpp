#include "ATNode.h"

#include "InstrStats.h"

#include "llvm/Support/raw_ostream.h"

// #include <iostream>
// #include <string>



// Constructor
ATNode :: ATNode (op_t operation) : op(operation), children(nullptr, nullptr) {}

// METHODS
void ATNode :: print() {

	printf("(");
	if(this->children.first != nullptr) this->children.first->print();
	printf("%s", this->op.to_string().c_str());
	if(this->children.second != nullptr) this->children.second->print();
	printf(")");
}

std::string ATNode :: to_string() {

	std::string node_string;

	node_string.append("(");
	if(this->children.first != nullptr) node_string.append(this->children.first->to_string());
	node_string.append(this->op.to_string());
	if(this->children.second != nullptr) node_string.append(this->children.second->to_string());
	node_string.append(")");

	return node_string;
}
