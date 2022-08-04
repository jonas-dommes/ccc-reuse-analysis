#ifndef ACCESSTREE_H
#define ACCESSTREE_H

#include "ATNode.h"

class AccessTree {

public:
	// DATA
	ATNode* root;

	// CONSTRUCTOR
	AccessTree();
	AccessTree(ATNode* rootNode);

	// METHODS
	void print();
	std::string to_string();
};



#endif
