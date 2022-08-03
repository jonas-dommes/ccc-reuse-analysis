#ifndef OPERATION_H
#define OPERATION_H

#include <string>

enum class op_t {
	ADD, SUB, MUL, DIV, REM, SHL, SHR, OR, AND, XOR, CALL, LOAD, PHI, GETELEPTR, UNDEF
};

class Operation {
public:

	op_t op;


	// CONSTRUCTOR
	Operation(op_t operation);

	// METHODS
	std::string to_string();
};



#endif
