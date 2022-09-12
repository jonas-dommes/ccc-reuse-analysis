#include "Offset.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"


#include <map>

using namespace llvm;


Offset :: Offset(int tid, int bid) : tid{tid}, bid{bid}, TidOffset {-1, -1, -1}, BidOffset {-1, -1, -1} {}


// Handle Ops
void Offset :: op_add(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] + b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] + b.BidOffset[i];
		}
	}
}

void Offset :: op_sub(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] - b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] - b.BidOffset[i];
		}
	}
}

void Offset :: op_mul(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] * b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] * b.BidOffset[i];
		}
	}
}

void Offset :: op_div(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] / b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] / b.BidOffset[i];
		}
	}
}

void Offset :: op_rem(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] % b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] % b.BidOffset[i];
		}
	}
}

void Offset :: op_shl(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] << b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] << b.BidOffset[i];
		}
	}
}

void Offset :: op_shr(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] >> b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] >> b.BidOffset[i];
		}
	}
}

void Offset :: op_or(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] | b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] | b.BidOffset[i];
		}
	}
}

void Offset :: op_and(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] & b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] & b.BidOffset[i];
		}
	}
}

void Offset :: op_xor(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		if (a.TidOffset[i] == -1 || b.TidOffset[i] == -1) {
			if (a.TidOffset[i] == -1) this->TidOffset[i] = b.TidOffset[i];
			if (b.TidOffset[i] == -1) this->TidOffset[i] = a.TidOffset[i];
		} else {
			this->TidOffset[i] = a.TidOffset[i] ^ b.TidOffset[i];
		}
		if (a.BidOffset[i] == -1 || b.BidOffset[i] == -1) {
			if (a.BidOffset[i] == -1) this->BidOffset[i] = b.BidOffset[i];
			if (b.BidOffset[i] == -1) this->BidOffset[i] = a.BidOffset[i];
		} else {
			this->BidOffset[i] = a.BidOffset[i] ^ b.BidOffset[i];
		}
	}
}

void Offset :: op_phi(Offset a, Offset b) {

	for (int i = 0; i < 3; i++) {
		this->TidOffset[i] = a.TidOffset[i];
		this->BidOffset[i] = a.BidOffset[i];
	}
}

void Offset :: op_pass_up(Offset a) {

	for (int i = 0; i < 3; i++) {
		this->TidOffset[i] = a.TidOffset[i];
		this->BidOffset[i] = a.BidOffset[i];
	}
}

// Handle Values
void Offset :: val_const_int(int val) {

	for (int i = 0; i < 3; i++) {
		this->TidOffset[i] = val;
		this->BidOffset[i] = val;
	}
}

void Offset :: val_cuda_reg(StringRef call_str) {

	std::map<char, unsigned int> char_map {{'x', 0}, {'y', 1}, {'z', 2}};
	std::pair<StringRef, StringRef> tmp = call_str.split('.');

	unsigned int dim = char_map[tmp.second.front()];

	if (tmp.first.equals("tid")) { // Thread Id

		this->TidOffset[dim] = this->tid;

	} else if (tmp.first.equals("ctaid")) { // Block Id

		this->BidOffset[dim] = this->bid;

	} else if (tmp.first.equals("ntid")) { // Block Dim

		if (dim <= 1 ) {
			this->BidOffset[dim] = 256;
		} else {
			this->BidOffset[dim] = 32;
		}

	} else if (tmp.first.equals("nctaid")) { // Grid Dim

		this->BidOffset[dim] = 1024;
	}
}

void Offset :: val_inc() {

	for (int i = 0; i < 3; i++) {
		this->TidOffset[i] = 0;
		this->BidOffset[i] = 0;
	}
}

void Offset :: val_arg() {

	for (int i = 0; i < 3; i++) {
		this->TidOffset[i] = 40;
		this->BidOffset[i] = 40;
	}
}

void Offset :: mul_by_dep(int* tid_dep, int* bid_dep) {

	for (int i = 0; i < 3; i++) {
		this->TidOffset[i] *= tid_dep[i];
		this->BidOffset[i] *= bid_dep[i];
	}
}


// Utlity
std::string Offset :: to_string() {

	std::string str = "\tTidOffset[";
	for (int& offset : this->TidOffset) {
		str.append(std::to_string(offset));
		str.append(", ");
	}
	str.pop_back();
	str.pop_back();

	str.append("]\n\tBidOffset[");
	for (int& offset : this->BidOffset) {
		str.append(std::to_string(offset));
		str.append(", ");
	}
	str.pop_back();
	str.pop_back();
	str.append("]\n");

	return str;
}

std::string Offset :: to_string_tid() {

	std::string str = "Id: ";
	str.append(std::to_string(this->tid));
	str.append("\tTidOffset[");
	for (int& offset : this->TidOffset) {
		str.append(std::to_string(offset));
		str.append(", ");
	}
	str.pop_back();
	str.pop_back();
	str.append("]");

	return str;
}

std::string Offset :: to_string_bid() {

	std::string str = "Id: ";
	str.append(std::to_string(this->bid));
	str.append("\tBidOffset[");
	for (int& offset : this->BidOffset) {
		str.append(std::to_string(offset));
		str.append(", ");
	}
	str.pop_back();
	str.pop_back();
	str.append("]");

	return str;
}
