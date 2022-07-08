#include <iostream>
#include <algorithm>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>

#include "../llvm-rpc-passes/Common.h"
#include "../llvm-rpc-passes/GridAnalysisPass.h"

#include "FunctionStats.h"

#include "InstrStats.h"

using namespace llvm;

// public:

void InstrStats::analyseInstr(Instruction *I, LoopInfo *LI, struct dependance_t dep_calls) {


	if (isa<StoreInst>(I)) {

		this->addr = I->getOperand(1);
		this->is_store = true;

	} else if (isa<LoadInst>(I)) {

		this->addr = I->getOperand(0);
		this->is_load = true;
	}

	this->getDataAlias(I);
	this->getLoopDepth(I, LI);
	this->isConditional(I);
	this->analyseDependence(I, dep_calls);
	this->analyseAccessPattern(I, dep_calls);
}


void InstrStats::printInstrStats() {

	if (this->is_load) {
		printf("\t\tLoad ");
	} else if (this->is_store) {
		printf("\t\tStore");
	}
	if (this->is_tid_dep) {
		printf("\tTID");
	}
	if (this->is_bid_dep) {
		printf("\tBID");
	}
	if (this->is_blocksize_dep) {
		printf("\tBSD");
	}
	if (this->is_gridsize_dep) {
		printf("\tGSD");
	}
	if (this->is_conditional) {
		printf("\tCOND");
	}
	printf("\n");

	printf("\t\tLoop Depth: %d\n", this->loop_depth);
	printf("\t\tAddr: %p\t\t Alias: %s\n", this->addr, this->data_alias.c_str());
	printf("\t\tAccess pattern: %s\n", this->access_pattern.c_str());

}


// private:

void InstrStats::getDataAlias(Instruction *I) {

	Instruction* data_instr;

	if (this->is_load) {
		data_instr = cast<Instruction>(I->getOperand(0));
		// errs() << *cast<Instruction>(I->getOperand(0)) << "\n";
	} else if (this->is_store) {
		data_instr = cast<Instruction>(I->getOperand(1));
	}
	// errs() << "Before while for " << *data_instr << "\n";

	// TODO: Handle PHI nodes
	while (isa<GetElementPtrInst>(data_instr) == false) {

		// errs() << "\tInstr has operand 0: " << *data_instr->getOperand(0) << "\n";
		// errs() << "\t\tOperand Type:      " << typeid(*data_instr->getOperand(0)).name() << "\n";

		if (isa<Instruction>(*data_instr->getOperand(0))) {

			data_instr = dyn_cast<Instruction>(data_instr->getOperand(0));
			// errs() << "\tdata_instr is now:   " << *data_instr << "\n";

		} else {
			this->data_alias = data_instr->getOperand(0)->getName();
			break;
		}
	}

	if (isa<GetElementPtrInst>(data_instr) == true) {
		// errs() << "Found getElemtPtr: " << *data_instr << "\n";
		this->data_alias = data_instr->getOperand(0)->getName();
	}

}


unsigned int InstrStats::getLoopDepth(Instruction *I, LoopInfo *LI) {

	this->loop_depth = LI->getLoopDepth((I->getParent()));

	return this->loop_depth;
}


void InstrStats::isConditional(llvm::Instruction *I) {

	std::string name = I->getParent()->getName();

	if (std::strncmp(name.c_str(), "if.", 3) == 0) {

		// errs() << I->getParent()->getName() << "\n";
		this->is_conditional = true;
	}

}


// TODO handle phis?
void InstrStats::analyseDependence(Instruction *I, struct dependance_t dep_calls) {

	std::set<Instruction*> workset;
	std::set<Instruction*> doneset;

	// errs() << "\nStarting Analysis of:\n\t" << *I << "\n";

	// Add Addr Operand of load/store instr to worklist
	if (this->is_load) {
		workset.insert(cast<Instruction>(I->getOperand(0)));
		// errs() << *cast<Instruction>(I->getOperand(0)) << "\n";
	} else if (this->is_store) {
		workset.insert(cast<Instruction>(I->getOperand(1)));
	}

	// while there are instr in workset
	while (!workset.empty()) {
		// get and delete first instr from workset, add to doneset
		Instruction* instr = *workset.begin();
		workset.erase(instr);
		doneset.insert(instr);

		// check if instr is in dep_calls
		if (dep_calls.tid_calls.count(instr) > 0) {

			this->is_tid_dep = true;
			// errs() << "Is in tid_calls: " << *instr << "\n";
		}
		if (dep_calls.bid_calls.count(instr) > 0) {

			this->is_bid_dep = true;
			// errs() << "Is in bid_calls: " << *instr << "\n";
		}
		if (dep_calls.blocksize_calls.count(instr) > 0) {

			this->is_blocksize_dep = true;
			// errs() << "Is in blocksize_calls: " << *instr << "\n";
		}
		if (dep_calls.gridsize_calls.count(instr) > 0) {

			this->is_gridsize_dep = true;
			// errs() << "Is in gridsize_calls: " << *instr << "\n";
		}

		// errs() << *instr << " has Operands:\n";

		// add operands to workset if not in doneset
		for (auto &op : instr->operands()) {
			Instruction* op_instr = dyn_cast<Instruction>(op);

			// Continue if Operand is no Instruction
			if (op_instr == NULL) {
				continue;
			}

			if (doneset.count(op_instr) == 0) {

				workset.insert(op_instr);
				// errs() << "\t" << *op_instr << "\n";
			}
		}
		// handle phis?
	}

	// if (!(this->is_tid_dep || this->is_bid_dep || this->is_blocksize_dep || this->is_gridsize_dep)) {
	// 	errs() << "\n-----------------Instruction was not relevant-----------------\n" << *I << "\n\n";
	// }
}

void InstrStats::analyseAccessPattern(llvm::Instruction *I, struct dependance_t dep_calls) {

	Instruction* data_instr;

	// Get correct starting Operand from load/store
	if (this->is_load) {
		data_instr = cast<Instruction>(I->getOperand(0));
	} else if (this->is_store) {
		data_instr = cast<Instruction>(I->getOperand(1));
	}

	errs() << "\nCreating access pattern for: " << *data_instr << "\n";

	// TODO: Handle PHI nodes
	// Run through instructions to find GetElementPtrInst
	while (isa<GetElementPtrInst>(data_instr) == false) {

		errs() << "\tInstr has operand 0: " << *data_instr->getOperand(0) << "\n";

		if (isa<Instruction>(*data_instr->getOperand(0))) {

			data_instr = dyn_cast<Instruction>(data_instr->getOperand(0));
			errs() << "\tdata_instr is now:   " << *data_instr << "\n";

		// If Access is at Start of array, no GetElementPtrInst gets generated
		} else if (isa<Argument>(*data_instr->getOperand(0))){

			this->access_pattern.append(data_instr->getOperand(0)->getName());
			this->access_pattern.append("[0]");
			errs() << "Resulting pattern: " << this->access_pattern << "\n";
			return;

		} else {
			errs() << "----------------NEVER REACHED BEFORE----------------\n";
			return;
		}
	}

	// Get Address Offset
	if (isa<GetElementPtrInst>(data_instr) == true) {
		errs() << "Found getElemtPtr: " << *data_instr << "\n";

		// Call visitOperand if offset is dynamic
		if (isa<Instruction>(data_instr->getOperand(1))) {

			data_instr = cast<Instruction>(data_instr->getOperand(1));
			errs() << "VisitOperand(" << *data_instr << ")\n" ;
			visitOperand(data_instr, dep_calls);

		// Retrieve constant value of offset
		} else if (isa<ConstantInt>(*data_instr->getOperand(1))) {

			ConstantInt *val = dyn_cast<ConstantInt>(data_instr->getOperand(1));
			this->access_pattern.append(data_instr->getOperand(0)->getName());
			this->access_pattern.append("[");
			this->access_pattern.append(std::to_string(val->getSExtValue()));
			this->access_pattern.append("]");
		}
		errs() << "Resulting pattern: " << this->access_pattern << "\n";
	}
}

// Handles access patterns of Operands, rekursiv
void InstrStats::visitOperand(llvm::Instruction *I, struct dependance_t dep_calls) {

	if (!isa<Instruction>(I)) {
		errs() << "Is Argument\n";
		return;
	}

	switch (I->getOpcode()) {
		case Instruction::Add:
			this->access_pattern.append("(");
			errs() << "VisitOperand(" << *I->getOperand(0) << ")\n" ;
			visitOperand(cast<Instruction>(I->getOperand(0)), dep_calls);
			this->access_pattern.append(" + ");
			errs() << "VisitOperand(" << *I->getOperand(1) << ")\n" ;
			visitOperand(cast<Instruction>(I->getOperand(1)), dep_calls);
			this->access_pattern.append(")");
			break;
		case Instruction::Sub:

			break;
		case Instruction::Mul:
		errs() << "VisitOperand(" << *I->getOperand(0) << ")\n" ;
			visitOperand(cast<Instruction>(I->getOperand(0)), dep_calls);
			this->access_pattern.append(" * ");
			errs() << "VisitOperand(" << *I->getOperand(1) << ")\n" ;
			visitOperand(cast<Instruction>(I->getOperand(1)), dep_calls);

			break;
		case Instruction::UDiv:

			break;
		case Instruction::SDiv:

			break;
		case Instruction::URem:

			break;
		case Instruction::SRem:

			break;
		case Instruction::Shl:

			break;
		case Instruction::LShr:

			break;
		case Instruction::AShr:

			break;
		case Instruction::Or:

			break;
		case Instruction::And:

			break;
		case Instruction::Xor:

			break;
		case Instruction::Call:
			// check if instr is in dep_calls
			if (dep_calls.tid_calls.count(I) > 0) {
				this->access_pattern.append("tid");
			}
			if (dep_calls.bid_calls.count(I) > 0) {
				this->access_pattern.append("bid");
			}
			if (dep_calls.blocksize_calls.count(I) > 0) {
				this->access_pattern.append("bDim");
			}
			if (dep_calls.gridsize_calls.count(I) > 0) {
				this->access_pattern.append("gDim");
			}

			break;
		case Instruction::Load:
			this->access_pattern.append("Depends on loaded value");
			break;
		case Instruction::PHI:
			this->access_pattern.append("Depends on Phi");
			break;
		default:
			errs() << "VisitOperand(" << *I->getOperand(0) << ")\n" ;
			visitOperand(cast<Instruction>(I->getOperand(0)), dep_calls);

			break;
	}
}
