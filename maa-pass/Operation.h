#ifndef OPERATION_H
#define OPERATION_H

#include <llvm/IR/Instructions.h>
#include "ATNode.h"

#include <string>
#include <vector>



using namespace llvm;

// enum class instr_t {
// 	NONE, ADD, SUB, MUL, DIV, REM, SHL, SHR, OR, AND, XOR, CALL, LOAD, STORE, PHI, GETELEPTR, OP_0, UNDEF
// };
// enum class val_t {
// 	NONE, ARG, CONST_INT, NAME, CUDA_REG
// };

class Operation {
public:

	instr_t op;
	std::vector<int> operands;


	// CONSTRUCTOR
	Operation(instr_t operation);
	Operation(Instruction* I);

	// METHODS
	std::string to_string();

private:
	void setOpFromInstr(Instruction* I);
	void initOperands();
};



#endif
