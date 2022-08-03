#include "Operation.h"

#include <string>

// CONSTRUCTOR
Operation :: Operation(op_t operation) : op(operation) {}

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
		case op_t::PHI: {
			op_string = " PHI{} ";
			break;
		}
		case op_t::GETELEPTR: {
			op_string = " GETELEMENTPTR ";
			break;
		}
		default:
		op_string = " DEFAULT OP ";
		break;
	}
	return op_string;
}
