#include "ATNode.h"

#include "InstrStats.h"

#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/Instructions.h>


// #include <iostream>
// #include <string>

using namespace llvm;


// Constructor
ATNode :: ATNode (Instruction* I) : op(I), instr(I), value(nullptr) {
	errs() << "Adding (" << this->op.operands.size() << ") nodes for " << *I << "\n";
	this->insert(I);
}

// METHODS
void ATNode :: insert(Instruction* I) {

	for (int i : op.operands) {
		if (Instruction* operand = dyn_cast<Instruction>(I->getOperand(i))) {

			errs() << "\tOperd: " << *operand << "\n";

			this->children.push_back(new ATNode(operand));
		} else if (Value* val = dyn_cast<Value>(I->getOperand(i))){

			errs() << "\tValue: " << *val << "\n";
			this->value = val;

		} else {
			errs() << "Not an instruction or value: "<< *I->getOperand(i) << "\n";
		}
	}
}



void ATNode :: print() {

	// errs() << this->op.to_string() << "(" << this->op.operands.size() << ")\n";
	//
	// for (auto& child : this->children) child->print();

	printf("%s", this->to_string().c_str());
	//
	// printf("(");
	// if(this->children.first != nullptr) this->children.first->print();
	// printf("%s", this->op.to_string().c_str());
	// if(this->children.second != nullptr) this->children.second->print();
	// printf(")");
	// if (this->op.operands.size() <= 1) {
	//
	// 	// printf("%s", this->op.to_string().c_str());
	// 	errs() <<  this->op.to_string();
	// 	// printf("(");
	// 	errs() << "(";
	// 	// if(this->children.size() == 1) node_string.append(this->children[0]->to_string());
	// 	if(this->children.size() == 1) errs() << this->children[0]->to_string();
	// 	// printf(")");
	// 	errs() << ")";
	//
	// }  else if (this->op.operands.size() == 2) {
	//
	// 	// node_string.append("(");
	// 	// if(this->children.size() == 1) node_string.append(this->children[0]->to_string());
	// 	// node_string.append(this->op.to_string());
	// 	// if(this->children.size() == 2) node_string.append(this->children[1]->to_string());
	// 	// node_string.append(")");
	//
	//
	// } else {
	// 	errs() << "[ATNODE::to_string()] should not have more than two Operands\n";
	// }
}

std::string ATNode :: to_string() {

	std::string node_string = "";

	if (this->op.operands.size() == 0) { //Should be Call

		if (this->op.op == op_t::CALL) {
			node_string.append(this->op.to_string());
		} else {
			errs() << "Zero Operands is no Call\n";
		}

	} else if (this->op.operands.size() == 1) {

		if (this->op.op == op_t::GETELEPTR) {

			node_string.append(this->instr->getOperand(0)->getName());
			node_string.append("[");

			if (ConstantInt* val = dyn_cast<ConstantInt>(this->instr->getOperand(1))) {
				node_string.append(std::to_string(val->getSExtValue()));
			} else if(this->children.size() == 1) {
				node_string.append(this->children[0]->to_string());
			}
			node_string.append("]");

		} else {

			if(this->children.size() == 1) {
				node_string.append(this->children[0]->to_string());
			} else {
				node_string.append(value_to_string());
			}
		}
	}  else if (this->op.operands.size() == 2) {

		if (this->op.op == op_t::PHI) {

			node_string.append("PHI");
			node_string.append("{");

			if(this->children.size() == 1) {
				node_string.append(this->children[0]->to_string());
			} else {
				node_string.append(value_to_string());
			}

			node_string.append(", ");

			if(this->children.size() == 2) {
				node_string.append(this->children[1]->to_string());
			} else {
				node_string.append(value_to_string());
			}

			node_string.append("}");

		} else {

			node_string.append("(");

			if(this->children.size() >= 1) {
				node_string.append(this->children[0]->to_string());
			} else {
				node_string.append(value_to_string());
			}

			node_string.append(this->op.to_string());

			if(this->children.size() >= 2) {
				node_string.append(this->children[1]->to_string());
			} else {
				node_string.append(value_to_string());
			}

			node_string.append(")");
		}
	} else {
		errs() << "[ATNODE::to_string()] should not have more than two Operands\n";
	}
	return node_string;
}

std::string ATNode :: value_to_string() {

	if (this->value == nullptr) {
		errs() << "Trying to get value from empty vector\n";
		return "";
	}

	std::string val_str = "";

	if (ConstantInt* const_val = dyn_cast<ConstantInt>(value)) {

		val_str.append(std::to_string(const_val->getSExtValue()));

	} else if (Argument* arg = dyn_cast<Argument>(value)) {

		val_str.append(arg->getName());
		val_str.append("[0]");

	} else {

		errs() << "[Printvalue] Neither Case matched\n";
	}
	return val_str;
}
