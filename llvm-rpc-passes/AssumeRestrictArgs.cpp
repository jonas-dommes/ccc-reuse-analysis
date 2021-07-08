#include "AssumeRestrictArgs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "Common.h"
#include "Util.h"

using namespace llvm;

extern cl::opt<std::string> CLKernelName;
extern cl::opt<std::string> CLCoarseningMode;

AssumeRestrictArgs::AssumeRestrictArgs() : FunctionPass(ID) {}

void AssumeRestrictArgs::getAnalysisUsage(AnalysisUsage &au) const {
    au.setPreservesCFG();
}

bool AssumeRestrictArgs::runOnFunction(Function &F) {
    std::string m_kernelName = CLKernelName;
    bool m_dynamicMode = CLCoarseningMode == "dynamic";
    bool hostCode = F.getParent()->getTargetTriple() != CUDA_TARGET_TRIPLE;
    
    if(hostCode || (!Util::shouldCoarsen(F, m_kernelName, hostCode, m_dynamicMode))) {
        return false;
    }

    bool isModified = false;
    for (Function::arg_iterator it = F.arg_begin(); it != F.arg_end(); ++it) {
        Argument& arg = *it;
        //errs() << "Arg is : " << arg << " " << arg.hasNoAliasAttr() << " ";
        if (arg.getType()->isPointerTy() && !arg.hasNoAliasAttr()) {
            //arg.addAttr(AttributeSet::get(arg.getContext(), AttributeSet::FunctionIndex, Attribute::NoAlias));
            arg.addAttr(Attribute::NoAlias);
            isModified = true;
        }
        //errs() << " now: " << arg.hasNoAliasAttr() << "\n";
    }

    return isModified;
}

char AssumeRestrictArgs::ID = 0;
static RegisterPass<AssumeRestrictArgs> X("assumeRestrictArgs", "Assume Restrict Args Pass - adds NoAlias attribute to pointer type function args");
