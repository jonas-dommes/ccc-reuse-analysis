#ifndef OFFSET_H
#define OFFSET_H

#include <string>

class Offset {
public:
// DATA
	int TidOffset[3];
	int BidOffset[3];

	// enum class instr_t {
	// 	NONE = 0, ADD, SUB, MUL, DIV, REM, SHL, SHR, OR, AND, XOR, CALL, LOAD, STORE, PHI, GEP, EXT
	// };
	//
	// enum class val_t {
	// 	NONE = 0, ARG, CONST_INT, CUDA_REG, INC
	// };

// METHODS
	Offset();

	// Handle Ops
	void op_add(Offset a, Offset b);
	void op_sub(Offset a, Offset b);
	void op_mul(Offset a, Offset b);
	void op_div(Offset a, Offset b);
	void op_rem(Offset a, Offset b);
	void op_shl(Offset a, Offset b);
	void op_shr(Offset a, Offset b);
	void op_or(Offset a, Offset b);
	void op_and(Offset a, Offset b);
	void op_xor(Offset a, Offset b);

	void op_call(Offset a);
	void op_phi(Offset a, Offset b);

	// Handle Values
	void val_const_int(int val);
	void val_cuda_reg(int* tid_dep, int* bid_dep);
	void val_inc();
	void val_arg();

	// Utility
	std::string to_string();
};


#endif
