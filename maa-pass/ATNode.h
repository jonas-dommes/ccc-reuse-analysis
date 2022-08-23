#ifndef ATNODE_H
#define ATNODE_H

#include "InstrStats.h"
#include "Offset.h"

#include <llvm/IR/Instructions.h>

#include <utility>

using namespace llvm;

enum class instr_t {
	NONE = 0, ADD, SUB, MUL, DIV, REM, SHL, SHR, OR, AND, XOR, CALL, LOAD, STORE, PHI, GEP, EXT
};

enum class val_t {
	NONE = 0, ARG, CONST_INT, CUDA_REG, INC
};

const std::string instr_t_str[] = {
	"NONE", "ADD", "SUB", "MUL", "DIV", "REM", "SHL", "SHR", "OR", "AND", "XOR", "CALL", "LOAD", "STORE", "PHI", "GEP", "EXT", "UNDEF"
};

const std::string val_t_str[] {
	"NONE", "ARG", "CONST_INT", "CUDA_REG", "INC"
};


class ATNode {
public:

	Value* value;
	InstrStats* instr_stats;
	ATNode* parent;
	std::vector<ATNode*> children;
	instr_t instr_type;
	val_t value_type;
	int int_val;
	StringRef name;
	int tid_dep[3];
	int bid_dep[3];
	Offset offset;

public:
	inline static std::set<Instruction*> visited_phis;

public:

	// Constructor
	ATNode (Value* value, InstrStats* instr_stats, ATNode* parent, val_t value_type, int int_val, StringRef name);
	ATNode(Value* value, InstrStats* instr_stats, ATNode* parent);

// METHODS
	void insertChildren(Instruction* I);
	void handleCallStr();
	void fillDims();
	void set_instr_type(Instruction* I);

	void calcOffset();
	void offsetValue();
	void offsetInstr();
	void offsetMulDep();
	void printErrsNode();
	std::string access_pattern_to_string();
	std::string access_pattern_instr();
	std::string access_pattern_value();
	bool isBinary();
	std::string op_to_string();
	std::string val_t_to_string();
	std::string instr_t_to_string();
	bool check_and_add_visited(Instruction *phi);

};


#endif
