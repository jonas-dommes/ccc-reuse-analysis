#include <string>
#include <set>
#include <iostream>
#include <map>

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <llvm/IR/Instructions.h>
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Analysis/LoopInfo.h>

#include "NVPTXUtilities.h"

#include "MemoryAccessAnalysis.h"
#include "PassStats.h"
#include "FunctionStats.h"
#include "InstrStats.h"
#include "Util.h"

using namespace llvm;


struct maa : public FunctionPass {

	static char ID;
	maa() : FunctionPass(ID) {}

	void getAnalysisUsage(AnalysisUsage &AU) const override {
		AU.setPreservesCFG();
		AU.addRequired<LoopInfoWrapperPass>();
	}


	bool runOnFunction(Function &F) override {

		std::set<Value *> load_addresses;
		std::set<Value *> store_addresses;
		std::map<Instruction*, InstrStats> instr_map;

		FunctionStats func_stats;
		func_stats.function_name = F.getName();

		func_stats.isKernel(F);

		// Stop if function is no kernelfunction
		if (!func_stats.is_kernel) {
			return false;
		}

		LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

		for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {

			// Only analyse store and load instructions, otherwise continue
			if (!isa<StoreInst>(*I) && !isa<LoadInst>(*I)) {
				continue;
			}

			InstrStats instr_stats;

			instr_stats.analyseInstr(&*I, &LI);



			if (isa<StoreInst>(*I)) {

				func_stats.num_stores++;
				store_addresses.insert(I->getOperand(1));


			} else if (isa<LoadInst>(*I)) {

				func_stats.num_loads++;
				load_addresses.insert(I->getOperand(0));
			}

			instr_map[&*I] = instr_stats;

		}


		func_stats.unique_loads = load_addresses.size();
		func_stats.unique_stores = store_addresses.size();

		// Get total unique loads and stores TODO proper addresses
		std::set<Value *> total;
		set_union(load_addresses.begin(), load_addresses.end(), store_addresses.begin(), store_addresses.end(), std::inserter(total, total.begin()));
		func_stats.unique_total = total.size();

		func_stats.instr_map = instr_map;

		func_stats.printFunctionStats();
		func_stats.printInstrMap();


		// errs() << stats.function_name.c_str() << "Kernel: " << isKernel << "\n";

		return false;
	}


private:

};




char maa::ID = 0;
static RegisterPass<maa> X("maa", "Memory Access Analysis Pass", true /* Only looks at CFG */, true /* Analysis Pass */);
