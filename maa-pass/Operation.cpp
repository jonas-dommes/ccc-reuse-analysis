#include "Operation.h"

#include <llvm/IR/Instructions.h>

#include <string>

using namespace llvm;

// CONSTRUCTOR
Operation :: Operation(op_t operation) : op(operation) {}

Operation :: Operation(Instruction* I) {

	this->setOpFromInstr(I);
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
			op_string = " CALL() ";
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
		}
		default: {
			this->op = op_t::UNDEF;
			break;
		}
	}
}
