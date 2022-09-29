#include "ATNode.h"

#include "InstrStats.h"
#include "Offset.h"

#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/Instructions.h>

#include <iostream>
#include <string>

using namespace llvm;


// Constructor
ATNode :: ATNode (Value* value, InstrStats* instr_stats, ATNode* parent, val_t value_type, int int_val, StringRef name) : value(value), instr_stats(instr_stats), parent(parent), instr_type(instr_t::NONE), value_type(value_type), int_val(int_val), name(name), tid_dep{0}, bid_dep{0} {}

ATNode :: ATNode (Value* value, InstrStats* instr_stats, ATNode* parent) : value(value), instr_stats(instr_stats), parent(parent), instr_type(instr_t::NONE), value_type(val_t::NONE), int_val{-1}, tid_dep{0}, bid_dep{0} {

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
		errs() << "[ATNode()] Is none of the above: " << *value << "\n";
	}

	// Pass up dependence
	if (this->parent != nullptr) {
		int i = 0;
		for (const int &dep : tid_dep) parent->tid_dep[i++] |= dep;
		i = 0;
		for (const int &dep : bid_dep) parent->bid_dep[i++] |= dep;
	}
	// printErrsNode();
}

// METHODS
void ATNode :: insertChildren(Instruction* I) {

	if (isa<StoreInst>(I)) {
		// errs() << "\tOp1" << ": " << *I->getOperand(1) << "\n";
		this->children.push_back(new ATNode(I->getOperand(1), this->instr_stats, this));
		return;
	}

	if (isa<SelectInst>(I)) {
		errs() << "\tAdding Children for select\n";
		errs() << "\tOp1" << ": " << *I->getOperand(1) << "\n";
		errs() << "\tOp2" << ": " << *I->getOperand(2) << "\n";
		this->children.push_back(new ATNode(I->getOperand(1), this->instr_stats, this));
		this->children.push_back(new ATNode(I->getOperand(2), this->instr_stats, this));
		return;
	}

	// // Print Operands
	// int i = 0;
	// for (Use& op : I->operands()) {
	// 	errs() << "\tOp" << i++ << ": " << *op << "\n";
	// }

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

	unsigned int dim = char_map[tmp.second.front()];

	if (tmp.first.equals("tid")) { // Thread Id

		if (dim > this->instr_stats->tid_dim) this->instr_stats->tid_dim = dim;
		this->tid_dep[dim - 1] = 1;

	} else if (tmp.first.equals("ctaid")) { // Block Id

		if (dim > this->instr_stats->bid_dim) this->instr_stats->bid_dim = dim;
		this->bid_dep[dim - 1] = 1;

	} else if (tmp.first.equals("ntid")) { // Block Dim

		if (dim > this->instr_stats->block_dim) this->instr_stats->block_dim = dim;

	} else if (tmp.first.equals("nctaid")) { // Grid Dim

		if (dim > this->instr_stats->grid_dim) this->instr_stats->grid_dim = dim;

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
		case Instruction::Select: {
			this->instr_type = instr_t::SEL;
			break;
		}
		case Instruction::Trunc:
		case Instruction::FPToUI:
		case Instruction::FPToSI:
		case Instruction::PtrToInt:
		case Instruction::IntToPtr:
		case Instruction::SExt:
		case Instruction::ZExt:
		case Instruction::AddrSpaceCast:
		case Instruction::BitCast: {
			this->instr_type = instr_t::EXT;
			break;
		}
		default: {
			errs() << "[set_instr_type()] Reached default case with" << *I << "\n";
			// Print operands
			int i = 0;
			for (Use& op : I->operands()) {
				errs() << "Op[" << i << "]: " << *op << "\n";
			}
			break;
		}
	}
}

void ATNode :: calcOffset() {

	this->offsets.push_back(new Offset(0, 0));
	this->offsets.push_back(new Offset(1, 1));
	this->offsets.push_back(new Offset(32, 8));

	if (this->instr_type == instr_t::NONE) {

		for (Offset* offset : this->offsets) {
			this->offsetValue(offset);
		}

	} else {

		for (ATNode* child : this->children) {
			child->calcOffset();
		}

		int i = 0;
		for (Offset* offset : this->offsets) {

			Offset* a = this->children.front()->offsets.at(i);
			Offset* b = this->children.back()->offsets.at(i);
			this->offsetInstr(this->offsets.at(i), a, b);
			i++;
		}
	}

	// errs() << *this->value << "\n" << this->offsets.front()->to_string();
	// errs() << this->offsets.back()->to_string();
}

void ATNode :: offsetValue(Offset* offset) {

	switch (this->value_type) {
		case val_t::CUDA_REG: {
			offset->val_cuda_reg(this->name);
			break;
		}
		case val_t::INC: {
			offset->val_inc();
			break;
		}
		case val_t::ARG: {
			offset->val_arg();
			break;
		}
		case val_t::CONST_INT: {
			offset->val_const_int(this->int_val);
			break;
		}
		default: {
			errs() << "No valid case in offsetValue(): " << this->val_t_to_string() << "\n";
			break;
		}
	}
}

void ATNode :: offsetInstr(Offset* out, Offset* a, Offset* b) {

	switch (this->instr_type) {
		case instr_t::ADD: {
			out->op_add(*a, *b);
			break;
		}
		case instr_t::SUB: {
			out->op_sub(*a, *b);
			break;
		}
		case instr_t::MUL: {
			out->op_mul(*a, *b);
			break;
		}
		case instr_t::DIV: {
			out->op_div(*a, *b);
			break;
		}
		case instr_t::REM: {
			out->op_rem(*a, *b);
			break;
		}
		case instr_t::SHL: {
			out->op_shl(*a, *b);
			break;
		}
		case instr_t::SHR: {
			out->op_shr(*a, *b);
			break;
		}
		case instr_t::OR: {
			out->op_or(*a, *b);
			break;
		}
		case instr_t::AND: {
			out->op_and(*a, *b);
			break;
		}
		case instr_t::XOR: {
			out->op_xor(*a, *b);
			break;
		}
		case instr_t::PHI: {
			out->op_phi(*a, *b);
			break;
		}
		// Pass up from only child:
		case instr_t::GEP: {
			out->op_pass_up(*b);
			break;
		}
		case instr_t::SEL: {
			out->op_sel(*a, *b);
			break;
		}
		case instr_t::EXT:
		case instr_t::LOAD:
		case instr_t::CALL:
		case instr_t::STORE: {
			out->op_pass_up(*a);
			break;
		}
		default: {
			errs() << "No valid case in offsetBinary(): " << this->instr_t_to_string() << "\n";
			break;
		}
	}
}

void ATNode :: offsetMulDep() {

	for (Offset* offset : this->offsets) {
		offset->mul_by_dep(this->tid_dep, this->bid_dep);
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

	errs() << "tid_dep: (";
	for (const int &dep : tid_dep) errs() << dep << " ";
	errs() << ") bid_dep: (";
	for (const int &dep : bid_dep) errs() << dep << " ";
	errs() << ")\n";

	errs() << this->offsets.front()->to_string();

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
		case instr_t::SEL: {
			str.append("SEL{");
			str.append(this->children[0]->access_pattern_to_string());
			str.append(",");
			str.append(this->children[1]->access_pattern_to_string());
			str.append("}");
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

bool ATNode :: isBinary() {

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
			return true;
		}
		case instr_t::PHI:
		case instr_t::GEP:
		case instr_t::SEL:
		case instr_t::EXT:
		case instr_t::LOAD:
		case instr_t::CALL:
		case instr_t::STORE:{
			break;
		}

		default: {
			errs() << "No valid case in access_pattern_instr()\n";
			break;
		}
	}
	return false;
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
