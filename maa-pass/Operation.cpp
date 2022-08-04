#include "Operation.h"

#include <llvm/IR/Instructions.h>

#include <string>
#include <vector>

using namespace llvm;

// CONSTRUCTOR
Operation :: Operation(op_t operation) : op(operation) {

	this->initOperands();
}

Operation :: Operation(Instruction* I) {

	this->setOpFromInstr(I);
	this->initOperands();
}

// METHODS
std::string Operation :: to_string() {

	std::string op_string;

	switch (this->op) {
		case op_t::ADD: {
			op_string = " + ";
			break;
		}
		case op_t::SUB: {
			op_string = " - ";
			break;
		}
		case op_t::MUL: {
			op_string = " * ";
			break;
		}
		case op_t::DIV: {
			op_string = " / ";
			break;
		}
		case op_t::REM: {
			op_string = " % ";
			break;
		}
		case op_t::SHL: {
			op_string = " << ";
			break;
		}
		case op_t::SHR: {
			op_string = " >> ";
			break;
		}
		case op_t::OR: {
			op_string = " | ";
			break;
		}
		case op_t::AND: {
			op_string = " & ";
			break;
		}
		case op_t::XOR: {
			op_string = " ^ ";
			break;
		}
		case op_t::CALL: {
			op_string = "CALL";
			break;
		}
		case op_t::LOAD: {
			op_string = " LOAD ";
			break;
		}
		case op_t::STORE: {
			op_string = " STORE ";
			break;
		}
		case op_t::PHI: {
			op_string = " PHI{} ";
			break;
		}
		case op_t::GETELEPTR: {
			op_string = " GETELEMENTPTR ";
			break;
		}
		case op_t::OP_0: {
			op_string = " OP_0 ";
			break;
		}
		default:
		op_string = " DEFAULT_OP ";
		break;
	}
	return op_string;
}

void Operation :: setOpFromInstr(Instruction* I) {

	switch (I->getOpcode()) {
		case Instruction::Add: {
			this->op = op_t::ADD;
			break;
		}
		case Instruction::Sub: {
			this->op = op_t::SUB;
			break;
		}
		case Instruction::Mul: {
			this->op = op_t::MUL;
			break;
		}
		case Instruction::UDiv:
		case Instruction::SDiv: {
			this->op = op_t::DIV;
			break;
		}
		case Instruction::URem:
		case Instruction::SRem: {
			this->op = op_t::REM;
			break;
		}
		case Instruction::Shl:
		case Instruction::LShr: {
			this->op = op_t::SHL;
			break;
		}
		case Instruction::AShr: {
			this->op = op_t::SHR;
			break;
		}
		case Instruction::Or: {
			this->op = op_t::OR;
			break;
		}
		case Instruction::And: {
			this->op = op_t::AND;
			break;
		}
		case Instruction::Xor: {
			this->op = op_t::XOR;
			break;
		}
		case Instruction::Call: {
			this->op = op_t::CALL;
			break;
		}
		case Instruction::Load: {
			this->op = op_t::LOAD;
			break;
		}
		case Instruction::Store: {
			this->op = op_t::STORE;
			break;
		}
		case Instruction::PHI: {
			this->op = op_t::PHI;
			break;
		}
		case Instruction::GetElementPtr: {
			this->op = op_t::GETELEPTR;
			break;
		}
		case Instruction::Trunc:
		case Instruction::SExt:
		case Instruction::ZExt:
		case Instruction::BitCast: {
			this->op = op_t::OP_0;
			break;
		}
		default: {
			this->op = op_t::UNDEF;
			errs() << "[setOpFromInstr()] Reached UNDEF with" << *I << "\n";
			break;
		}
	}
}

void Operation :: initOperands() {

	switch (this->op) {
		case op_t::ADD:
		case op_t::SUB:
		case op_t::MUL:
		case op_t::DIV:
		case op_t::REM:
		case op_t::SHL:
		case op_t::SHR:
		case op_t::OR:
		case op_t::AND:
		case op_t::XOR:{
			this->operands.push_back(0);
			this->operands.push_back(1);
			break;
		}
		case op_t::CALL: {
			break;
		}
		case op_t::OP_0:
		case op_t::LOAD: {
			this->operands.push_back(0);
			break;
		}
		case op_t::GETELEPTR:
		case op_t::STORE: {
			this->operands.push_back(1);
			break;
		}
		case op_t::PHI: {
			this->operands.push_back(0);
			this->operands.push_back(1);
			break;
		}
		default:
		 	errs() << "[initOperands()] Reached DEFAULT_OP\n";
		break;
	}
}
