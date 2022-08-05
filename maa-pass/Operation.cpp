#include "Operation.h"

#include <llvm/IR/Instructions.h>

#include <string>
#include <vector>

using namespace llvm;

// CONSTRUCTOR
Operation :: Operation(instr_t operation) : op(operation) {

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
		case instr_t::ADD: {
			op_string = " + ";
			break;
		}
		case instr_t::SUB: {
			op_string = " - ";
			break;
		}
		case instr_t::MUL: {
			op_string = " * ";
			break;
		}
		case instr_t::DIV: {
			op_string = " / ";
			break;
		}
		case instr_t::REM: {
			op_string = " % ";
			break;
		}
		case instr_t::SHL: {
			op_string = " << ";
			break;
		}
		case instr_t::SHR: {
			op_string = " >> ";
			break;
		}
		case instr_t::OR: {
			op_string = " | ";
			break;
		}
		case instr_t::AND: {
			op_string = " & ";
			break;
		}
		case instr_t::XOR: {
			op_string = " ^ ";
			break;
		}
		case instr_t::CALL: {
			op_string = "CALL";
			break;
		}
		case instr_t::LOAD: {
			op_string = " LOAD ";
			break;
		}
		case instr_t::STORE: {
			op_string = " STORE ";
			break;
		}
		case instr_t::PHI: {
			op_string = " PHI{} ";
			break;
		}
		case instr_t::GETELEPTR: {
			op_string = " GETELEMENTPTR ";
			break;
		}
		case instr_t::EXT: {
			op_string = " EXT ";
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
			this->op = instr_t::ADD;
			break;
		}
		case Instruction::Sub: {
			this->op = instr_t::SUB;
			break;
		}
		case Instruction::Mul: {
			this->op = instr_t::MUL;
			break;
		}
		case Instruction::UDiv:
		case Instruction::SDiv: {
			this->op = instr_t::DIV;
			break;
		}
		case Instruction::URem:
		case Instruction::SRem: {
			this->op = instr_t::REM;
			break;
		}
		case Instruction::Shl:
		case Instruction::LShr: {
			this->op = instr_t::SHL;
			break;
		}
		case Instruction::AShr: {
			this->op = instr_t::SHR;
			break;
		}
		case Instruction::Or: {
			this->op = instr_t::OR;
			break;
		}
		case Instruction::And: {
			this->op = instr_t::AND;
			break;
		}
		case Instruction::Xor: {
			this->op = instr_t::XOR;
			break;
		}
		case Instruction::Call: {
			this->op = instr_t::CALL;
			break;
		}
		case Instruction::Load: {
			this->op = instr_t::LOAD;
			break;
		}
		case Instruction::Store: {
			this->op = instr_t::STORE;
			break;
		}
		case Instruction::PHI: {
			this->op = instr_t::PHI;
			break;
		}
		case Instruction::GetElementPtr: {
			this->op = instr_t::GETELEPTR;
			break;
		}
		case Instruction::Trunc:
		case Instruction::SExt:
		case Instruction::ZExt:
		case Instruction::BitCast: {
			this->op = instr_t::EXT;
			break;
		}
		default: {
			errs() << "[setOpFromInstr()] Reached UNDEF with" << *I << "\n";
			break;
		}
	}
}

void Operation :: initOperands() {

	switch (this->op) {
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
			this->operands.push_back(0);
			this->operands.push_back(1);
			break;
		}
		case instr_t::CALL: {
			break;
		}
		case instr_t::EXT:
		case instr_t::LOAD: {
			this->operands.push_back(0);
			break;
		}
		case instr_t::GETELEPTR:
		case instr_t::STORE: {
			this->operands.push_back(1);
			break;
		}
		case instr_t::PHI: {
			this->operands.push_back(0);
			this->operands.push_back(1);
			break;
		}
		default:
		 	errs() << "[initOperands()] Reached DEFAULT_OP\n";
		break;
	}
}
