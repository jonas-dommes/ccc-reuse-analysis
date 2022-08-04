#ifndef OPERATION_H
#define OPERATION_H

#include <llvm/IR/Instructions.h>

#include <string>
#include <vector>



using namespace llvm;

enum class op_t {
	ADD, SUB, MUL, DIV, REM, SHL, SHR, OR, AND, XOR, CALL, LOAD, STORE, PHI, GETELEPTR, UNDEF
};

class Operation {
public:

	op_t op;
	std::vector<int> operands;


	// CONSTRUCTOR
	Operation(op_t operation);
	Operation(Instruction* I);

	// METHODS
	std::string to_string();

private:
	void setOpFromInstr(Instruction* I);
	void initOperands();
};



#endif
