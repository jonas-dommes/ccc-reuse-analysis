#include "ATNode.h"

#include "InstrStats.h"

#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/Instructions.h>

#include <iostream>
#include <string>

using namespace llvm;


// Constructor
ATNode :: ATNode (Value* value, InstrStats* instr_stats, ATNode* parent, val_t value_type, int int_val, StringRef name) : value(value), instr_stats(instr_stats), parent(parent), instr_type(instr_t::NONE), value_type(value_type), int_val(int_val), name(name) {}

ATNode :: ATNode (Value* value, InstrStats* instr_stats, ATNode* parent) : value(value), instr_stats(instr_stats), parent(parent), instr_type(instr_t::NONE), value_type(val_t::NONE), int_val{-1} {

	if (Instruction* I = dyn_cast<Instruction>(value)) {

		errs() << "Insert Children of" << *I << "\n";

		this->set_instr_type(I);
		this->insertChildren(I);

	} else if (Argument* arg = dyn_cast<Argument>(value) ){

		this->value_type = val_t::ARG;
		this->name = arg->getName();
		errs() << "Added Argument with name " << this->name << "\n";


	} else if (ConstantInt* const_int = dyn_cast<ConstantInt>(value) ){

		this->value_type = val_t::CONST_INT;
		this->int_val = const_int->getSExtValue ();
		errs() << "Added ConstantInt with value " << this->int_val << "\n";

	} else if (CallInst* call = dyn_cast<CallInst>(this->parent->value) ){

		this->value_type = val_t::CUDA_REG;
		this->handleCallStr();
		errs() << "Found CallInst with Functionname: " << this->name << "\n";

	} else {
		errs() << "Is none of the above: " << *value << "\n";
	}


	// printErrsNode();
}

// METHODS
void ATNode :: insertChildren(Instruction* I) {

	if (isa<StoreInst>(I)) {
		errs() << "\tOp1" << ": " << *I->getOperand(1) << "\n";
		this->children.push_back(new ATNode(I->getOperand(1), this->instr_stats, this));
		return;
	}

	// Print Operands
	int i = 0;
	for (Use& op : I->operands()) {
		errs() << "\tOp" << i++ << ": " << *op << "\n";
	}

	for (Use& op : I->operands()) {

		if (Instruction* tmp = dyn_cast<Instruction>(op)) {

			if (isa<PHINode>(tmp) && check_and_add_visited(tmp)) {
				this->children.push_back(new ATNode(op, this->instr_stats, this, val_t::INC, -1, "INC"));
				continue;
			}
		}
		this->children.push_back(new ATNode(op, this->instr_stats, this));
	}

	// If num ops != num children --> error
	if (I->getNumOperands() != this->children.size()) {
		errs() << "[insertChilrdren()] Num Children does not match number of operands\n";
	}
}

void ATNode :: handleCallStr() {

	StringRef call_name = this->value->getName();
	StringRef prefix = "llvm.nvvm.read.ptx.sreg.";

	call_name.consume_front(prefix);
	this->name = call_name;

	this->fillDims();
}

void ATNode :: fillDims() {

	std::map<char, unsigned int> char_map {{'x', 1}, {'y', 2}, {'z', 3}};

	std::pair<StringRef, StringRef> tmp = this->name.split('.');

	if (tmp.first.equals("tid")) { // Thread Id

		if (char_map[tmp.second.front()] > this->instr_stats->tid_dim) this->instr_stats->tid_dim = char_map[tmp.second.front()];

	} else if (tmp.first.equals("ctaid")) { // Block Id

		if (char_map[tmp.second.front()] > this->instr_stats->bid_dim) this->instr_stats->bid_dim = char_map[tmp.second.front()];

	} else if (tmp.first.equals("ntid")) { // Block Dim

		if (char_map[tmp.second.front()] > this->instr_stats->block_dim) this->instr_stats->block_dim = char_map[tmp.second.front()];

	} else if (tmp.first.equals("nctaid")) { // Grid Dim

		if (char_map[tmp.second.front()] > this->instr_stats->grid_dim) this->instr_stats->grid_dim = char_map[tmp.second.front()];

	}
}

void ATNode :: set_instr_type(Instruction* I) {

	switch (I->getOpcode()) {
		case Instruction::Add: {
			this->instr_type = instr_t::ADD;
			break;
		}
		case Instruction::Sub: {
			this->instr_type = instr_t::SUB;
			break;
		}
		case Instruction::Mul: {
			this->instr_type = instr_t::MUL;
			break;
		}
		case Instruction::UDiv:
		case Instruction::SDiv: {
			this->instr_type = instr_t::DIV;
			break;
		}
		case Instruction::URem:
		case Instruction::SRem: {
			this->instr_type = instr_t::REM;
			break;
		}
		case Instruction::Shl:
		case Instruction::LShr: {
			this->instr_type = instr_t::SHL;
			break;
		}
		case Instruction::AShr: {
			this->instr_type = instr_t::SHR;
			break;
		}
		case Instruction::Or: {
			this->instr_type = instr_t::OR;
			break;
		}
		case Instruction::And: {
			this->instr_type = instr_t::AND;
			break;
		}
		case Instruction::Xor: {
			this->instr_type = instr_t::XOR;
			break;
		}
		case Instruction::Call: {
			this->instr_type = instr_t::CALL;
			break;
		}
		case Instruction::Load: {
			this->instr_type = instr_t::LOAD;
			break;
		}
		case Instruction::Store: {
			this->instr_type = instr_t::STORE;
			break;
		}
		case Instruction::PHI: {
			this->instr_type = instr_t::PHI;
			break;
		}
		case Instruction::GetElementPtr: {
			this->instr_type = instr_t::GEP;
			break;
		}
		case Instruction::Trunc:
		case Instruction::FPToUI:
		case Instruction::FPToSI:
		case Instruction::PtrToInt:
		case Instruction::IntToPtr:
		case Instruction::SExt:
		case Instruction::ZExt:
		case Instruction::BitCast: {
			this->instr_type = instr_t::EXT;
			break;
		}
		default: {
			errs() << "[set_instr_type()] Reached default case with" << *I << "\n";
			break;
		}
	}
}

void ATNode :: printErrsNode() {

	errs() << "\n============== NODE:" << *this->value << " ============== \n";

	if(this->parent != nullptr) errs() << "Parent:" << *this->parent->value << "\n";

	int i = 0;
	for (ATNode* val : this->children) {
		errs() << "Child" << i++ << ":" << *val->value << "\n";
	}

	errs() << "instr_t: " << this->instr_t_to_string() << " val_t: " << this->val_t_to_string() << "\n";

	if (value_type != val_t::NONE)  {
		if (value_type == val_t::CONST_INT) {
			errs() << "int_val " << this->int_val << "\n";
		} else {
			errs() << "name: " << this->name << "\n";
		}
	}

	errs() << "============== NODE END ============== \n";
}

std::string ATNode :: access_pattern_to_string() {

	std::string str = "";
	visited_phis.clear();

	if (value_type != val_t::NONE) {

		str.append(access_pattern_value());

	} else if (instr_type != instr_t::NONE) {

		str.append(access_pattern_instr());

	} else {
		errs() << "Neither value_type nor instr_type are set \n";
	}

	return str;
}

std::string ATNode :: access_pattern_instr() {

	std::string str = "";

	switch (this->instr_type) {
		case instr_t::ADD:
		case instr_t::SUB:
		case instr_t::MUL:
		case instr_t::DIV:
		case instr_t::REM:
		case instr_t::SHL:
		case instr_t::SHR:
		case instr_t::OR:
		case instr_t::AND:
		case instr_t::XOR:{
			str.append("(");
			str.append(this->children[0]->access_pattern_to_string());
			str.append(this->op_to_string());
			str.append(this->children[1]->access_pattern_to_string());
			str.append(")");
			break;
		}
		case instr_t::PHI: {
			if (check_and_add_visited(cast<Instruction>(this->value)) == false) {

				str.append("PHI{");
				str.append(this->children[0]->access_pattern_to_string());
				str.append(this->op_to_string());
				str.append(this->children[1]->access_pattern_to_string());
				str.append("}");
			}
			break;
		}
		case instr_t::GEP: {
			str.append(this->children[0]->access_pattern_to_string());
			str.append("[");
			str.append(this->children[1]->access_pattern_to_string());
			str.append("]");
			break;
		}
		case instr_t::EXT:
		case instr_t::LOAD:
		case instr_t::CALL: {
			str.append(this->children[0]->access_pattern_to_string());
			break;
		}
		case instr_t::STORE:{
			str.append(this->children[0]->access_pattern_to_string());
			break;
		}
		default: {
			errs() << "No valid case in access_pattern_instr()\n";
			break;
		}
	}

	return str;
}

std::string ATNode :: access_pattern_value() {

	std::string str = "";

	switch (this->value_type) {
		case val_t::CUDA_REG:
		case val_t::INC:
		case val_t::ARG: {
			str.append(this->name);
			break;
		}
		case val_t::CONST_INT: {
			str.append(std::to_string(this->int_val));
			break;
		}
		default: {
			errs() << "No valid case in access_patter_value(): " << *this->parent->value << "\n";
			break;
		}
	}

	return str;
}

std::string ATNode :: op_to_string() {

	std::string str = "";

	switch (this->instr_type) {
		case instr_t::ADD: {
			str = " + ";
			break;
		}
		case instr_t::SUB: {
			str = " - ";
			break;
		}
		case instr_t::MUL: {
			str = " * ";
			break;
		}
		case instr_t::DIV: {
			str = " / ";
			break;
		}
		case instr_t::REM: {
			str = " % ";
			break;
		}
		case instr_t::SHL: {
			str = " << ";
			break;
		}
		case instr_t::SHR: {
			str = " >> ";
			break;
		}
		case instr_t::OR: {
			str = " | ";
			break;
		}
		case instr_t::AND: {
			str = " & ";
			break;
		}
		case instr_t::XOR: {
			str = " ^ ";
			break;
		}
		case instr_t::PHI: {
			str = ", ";
			break;
		}
		default:
		break;
	}
	return str;
}

std::string ATNode :: val_t_to_string() {

	return val_t_str[static_cast<int>(this->value_type)];
}

std::string ATNode :: instr_t_to_string() {

	return instr_t_str[static_cast<int>(this->instr_type)];
}

bool ATNode :: check_and_add_visited(Instruction *phi) {

	if (this->visited_phis.count(phi) == 0) {

		this->visited_phis.insert(phi);
		return false; // phi was visited for first time

	}

	return true; // phi was visited before
}
