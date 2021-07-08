#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include <set>
#include <map>
#include <list>
#include <functional>
#include <algorithm>

#include "Common.h"
#include "Util.h"
#include "DivergenceAnalysisPass.h"
#include "GridAnalysisPass.h"
#include "CacheLineReuseAnalysis.h"

#define DEBUG_PRINT

using namespace llvm;

const int MAX_DIMENSIONS[] = {32,2,2};

extern cl::opt<std::string> CLKernelName;
extern cl::opt<std::string> CLCoarseningMode;
cl::opt<unsigned int> WarpSize("warp-size", cl::init(32), cl::Hidden, cl::desc("The size of one warp within which threads perform lock-step execution"));
cl::opt<unsigned int> CacheLineSize("cache-line-size", cl::init(32), cl::Hidden, cl::desc("The size of a cache line in bytes"));

void CacheLineReuseAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<GridAnalysisPass>();
}

bool CacheLineReuseAnalysis::runOnFunction(Function &F) {
    // Checking if this function should be analysed
    m_kernelName = CLKernelName;
    m_dynamicMode = CLCoarseningMode == "dynamic";
    bool hostCode = F.getParent()->getTargetTriple() != CUDA_TARGET_TRIPLE;
    
    if(hostCode || (!Util::shouldCoarsen(F, m_kernelName, hostCode, m_dynamicMode))) {
        return false;
    }

    // Initialisations
    memops.clear();
    relevantInstructions.clear();
    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_postDomT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    m_domT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    m_gridAnalysis = &getAnalysis<GridAnalysisPass>();
    dimensions = getDimensionality();

    #ifdef DEBUG_PRINT
        errs() << "Kernel ";
        errs().write_escaped(F.getName()) << " is " << dimensions << "-dimensional\n";
        errs() << "Args:\n";
        for (Argument &A : F.args()) {
            errs() << "  -- " << A << "\n";
        }
    #endif

    preprocess(F, memops, relevantInstructions);

    #ifdef DEBUG_PRINT
        errs() << "Printing memops:\n";
        for (Instruction * memop : memops) {
          errs() << *memop << "\n";
        }
        errs() << "There are " << relevantInstructions.size() << " relevant instructions\n";
  
        // do a bit of printing 
        for (BasicBlock &B : F) {
            errs() << "Basic block " << B.getName() << " has " << B.size() << " instructions\n";
            for (Instruction &I: B) {
                errs() << (relevantInstructions.count(&I) ? ">>" : "  ") << I << "\n";
            }
        }
    #endif

    
    worklist.push_back(&*F.begin());
    while(!worklist.empty()) {
        BasicBlock *block = *worklist.begin();
        worklist.erase(worklist.begin());
        std::map<Instruction*, std::vector<MemAccessDescriptor>> ins;
        INs[block] = ins;
        for (BasicBlock *pred : predecessors(block)) {
            // TODO merge-copy OUTs[pred] into INs
        }
        OUTs[block] = INs[block]; // using OUTs as the working set

        visitBB(block);
        for (BasicBlock *succ : successors(block)) {
            worklist.push_back(succ);
            // TODO handle loops
        }
    }

    // simulate memory accesses for all in memops
    simulateMemoryAccesses();
  
    return false;
}

void CacheLineReuseAnalysis::preprocess(Function &F, std::set<Instruction*>& memops, std::set<Instruction*>& relevantInstructions) {
    std::set<Instruction*> defs;
    std::set<BasicBlock*> relevantBlocks;
    // Step 1. Find all cached memory accesses
    for (BasicBlock &B : F) {
        #ifdef DEBUG_PRINT
            errs() << "Basic block " << B.getName() << " has " << B.size() << " instructions \n";
        #endif
        for (Instruction &I: B) {
            if ((I.getOpcode() == Instruction::Load || I.getOpcode() == Instruction::Store) && isCachedMemoryAccess(&I)) {

                // Step 1.a Fetch the memory access
                memops.insert(&I);

                const int paramIdx = I.getOpcode() == Instruction::Load ? 0 : 1;
            	if (Instruction *def = dyn_cast<Instruction>(I.getOperand(paramIdx))) {
                    // Step 1.b Find the definition of the accessed value and insert into defs
                    //while (def->getOpcode() == Instruction::BitCast) {
                    //    def = dyn_cast<Instruction>(def->getOperand(0));
                    //}
                    defs.insert(def);
                    #ifdef DEBUG_PRINT
                        errs () << "preprocessed " << *def << "\n";
                    #endif

                    // Step 1.c Initialise empty map for later use
                    // This logic requires -fno-discard-value-names
                    StringRef symbolName = getAccessedSymbolName(def);;
                    if (!symbolName.empty()) {
                        accessedCacheLines.insert(std::pair<StringRef, std::set<int>>(symbolName, std::set<int>()));
                        relevantBlocks.insert(&B);
                    }
                }
            }
	    }
    }

    // Step 2. Deep traverse of the use-def chain to find all relevant definitions
    while (defs.size() > 0) {
        Instruction *inst = *defs.begin();
        defs.erase(defs.begin());
        // avoid looping infinitely
        if (relevantInstructions.count(inst) == 0) {
            relevantInstructions.insert(inst);
            relevantBlocks.insert(inst->getParent());
            for (Use &u : inst->operands()) {
                if (Instruction *operand = dyn_cast<Instruction>(u.get())) {
                    defs.insert(operand);
                }
            }
        }
    }
}

void CacheLineReuseAnalysis::simulateMemoryAccesses() {
    for (Instruction *inst : memops) {
        #ifdef DEBUG_PRINT
            errs() << "Simulating mem access for " << *inst << "\n";
        #endif
        int alignment = -1;
        bool isStore = false;
        if (inst->getOpcode() == Instruction::Load) {
            alignment = dyn_cast<LoadInst>(inst)->getAlignment();
        } else if (inst->getOpcode() == Instruction::Store) {
            alignment = dyn_cast<StoreInst>(inst)->getAlignment();
            isStore = true;
        }
        Value * ptr = getAccessedSymbolPtr(inst);
        StringRef accessedSymbolName = getAccessedSymbolName(ptr);
        std::vector<MemAccessDescriptor> mads = getMADs(ptr);
        std::set<int> * prevAccesses = &accessedCacheLines[accessedSymbolName];
        std::set<int> accessesToAdd; 
        for (MemAccessDescriptor & mad : mads) {
            mad.print();
            bool fullCoalescing = true;
            list<int> accesses = mad.getMemAccesses(WarpSize, alignment, CacheLineSize, &fullCoalescing);
            int accessesNum = accesses.size();
            accesses.sort();
            accesses.unique();
            int uniqueAccessesNum = accesses.size();
            int duplicates = accessesNum - uniqueAccessesNum;
            
            if (duplicates == 0 && uniqueAccessesNum > 1) {
                std::vector<int> intersection(prevAccesses->size() + accesses.size());
                duplicates = set_intersection(prevAccesses->begin(), prevAccesses->end(), accesses.begin(), accesses.end(), intersection.begin()) - intersection.begin();
            }
            
            accessesToAdd.insert(accesses.begin(), accesses.end());
            
            #ifdef DEBUG_PRINT
                errs() << "Returned " << accessesNum << " accesses to " << uniqueAccessesNum << " unique cache lines with " << duplicates << " duplicates\n";
            #endif
            if (isStore && fullCoalescing) {
                errs() << "Ignoring mem accesses of fully coalesced store instruction";
            } else if (duplicates > 0 && uniqueAccessesNum > 1) {
                errs() << "Cache line re-use in access to [" << std::string(accessedSymbolName) << "]";
            }
        }
        prevAccesses->insert(accessesToAdd.begin(), accessesToAdd.end());
    }

}

// ********************************************************************************************* //
//                                         Utility functions                                     //
// ********************************************************************************************* //

Value * CacheLineReuseAnalysis::getAccessedSymbolPtr(Value * v) {
    // returns what holds the pointer (e.g. GetElementPtr or pointer passed as function arg) to the accessed symbol
    if (Instruction * inst = dyn_cast<Instruction>(v)) {
        switch (inst->getOpcode()) {
            case Instruction::Load:
                return getAccessedSymbolPtr(inst->getOperand(0));
            case Instruction::Store:
                return getAccessedSymbolPtr(inst->getOperand(1));
            case Instruction::BitCast:
                return getAccessedSymbolPtr(inst->getOperand(0));
            case Instruction::GetElementPtr:
                return inst;
            default:
                errs() << "ERROR: Could not cast operand to any known type (opcode: " << std::to_string(inst->getOpcode()) << ")\n";
                return v;
        }
    } else if (isa<Argument>(v)) {
        return v;
    }
    errs() << "ERROR: No case for retreiving symbol ptr for " << std::string(v->getName()) << "\n";
    return v;
}

StringRef CacheLineReuseAnalysis::getAccessedSymbolName(Value * v) {
    Value * ptr = getAccessedSymbolPtr(v);
    if (GetElementPtrInst * gep = dyn_cast<GetElementPtrInst>(ptr)) {
        return gep->getOperand(0)->getName();
    } else if (isa<Argument>(v)) {
        return v->getName();
    }
    errs() << "ERROR: Failed to retrieve symbol name of " << *v  << " (name: " << v->getName() << ")\n";
    return "";
}

inline bool isCachedAddressSpace(unsigned int addressSpace) {
    return addressSpace == ADDRESS_SPACE_DEFAULT || addressSpace == ADDRESS_SPACE_GLOBAL;
}

bool CacheLineReuseAnalysis::isCachedMemoryAccess(Instruction * inst) {
    // For convenience, this takes load/store/GEP instructions
    switch(inst->getOpcode()) {
        case Instruction::Load:
            return isCachedAddressSpace(dyn_cast<LoadInst>(inst)->getPointerAddressSpace());
        case Instruction::Store:
            return isCachedAddressSpace(dyn_cast<StoreInst>(inst)->getPointerAddressSpace());
        case Instruction::GetElementPtr:
            return isCachedAddressSpace(dyn_cast<GetElementPtrInst>(inst)->getPointerAddressSpace());
        default:
            errs() << "ERROR: Could not obtain address space of unknown instruction type (opcode: " << std::to_string(inst->getOpcode()) << ")";
            return false;
    }
}

int CacheLineReuseAnalysis::getDimensionality() {
    // returns the number of dimensions, e.g. whether kernel uses x, (x,y), (x,y,z) dimensions
    if (!m_gridAnalysis->getThreadIDDependentInstructions(2).empty()) {
        return 3;
    } else if (!m_gridAnalysis->getThreadIDDependentInstructions(1).empty()) {
        return 2;
    }
    return 1;
}

// ********************************************************************************************* //
//                                         Visitor functions                                     //
// ********************************************************************************************* //

void CacheLineReuseAnalysis::visitBB(BasicBlock *block) {
    for (Instruction &I : *block) {
        if (relevantInstructions.find(&I) != relevantInstructions.end()) {
            visitInst(&I);
            // will print values of MADs
            //if (CallInst *c = dyn_cast<CallInst>(&I)) {
            //    errs() << c->getCalledFunction()->getName() << ": ";
            //} else {
            //    errs() << I.getName() << ": ";
            //}
            //(*(OUTs[block][&I].begin())).print();
        }
    }
}

void CacheLineReuseAnalysis::visitInst(Instruction *inst) {
    switch (inst->getOpcode()) {
        case Instruction::Call          :       visitCall(dyn_cast<CallInst>(inst));                      break;
        case Instruction::Add           :       visitBinaryOp(std::plus<int>(), inst);                    break;
        case Instruction::Sub           :       visitBinaryOp(std::minus<int>(), inst);                   break;
        case Instruction::Mul           :       visitBinaryOp(std::multiplies<int>(), inst);              break;
        case Instruction::UDiv          :       visitBinaryOp(std::divides<int>(), inst);                 break;
        case Instruction::URem          :       visitBinaryOp(std::modulus<int>(), inst);                 break;
        case Instruction::Shl           :       visitBinaryOp([](int a, int b) {return a << b;}, inst);   break;
        case Instruction::Or            :       visitBinaryOp(std::bit_or<int>(), inst);                  break;
        case Instruction::And           :       visitBinaryOp(std::bit_and<int>(), inst);                 break;
        case Instruction::Xor           :       visitBinaryOp(std::bit_xor<int>(), inst);                 break;
        case Instruction::SExt          :       visitSExt(inst);                                          break;
        case Instruction::Trunc         :       visitTrunc(inst);                                         break;
        case Instruction::ICmp          :       visitICmp(dyn_cast<ICmpInst>(inst));                      break;
        //case Instruction::Select        :       visitSelect() //TODO
        case Instruction::BitCast       :       visitBitCast(inst);                                       break;
        case Instruction::GetElementPtr :       visitGetElementPtr(inst);                                 break;
        default: errs() << "ERROR: Unsupported opcode for instruction " << *inst << "\n";
    }
}


// TODO
//#define CUDA_SHUFFLE_DOWN      "nvvm.shfl.down"
//#define CUDA_SHUFFLE_UP        "nvvm.shfl.up"
//#define CUDA_SHUFFLE_BFLY      "nvvm.shfl.bfly"
//#define CUDA_SHUFFLE_IDX       "nvvm.shfl.idx"

void CacheLineReuseAnalysis::visitCall(CallInst *call) {
    Function *callee = call->getCalledFunction();
    StringRef calleeN = callee->getName();
    std::string prefix = LLVM_PREFIX;
    prefix.append(".");
    prefix.append(CUDA_READ_SPECIAL_REG);
    prefix.append(".");
    if      ( calleeN.startswith(prefix + CUDA_THREAD_ID_REG )) { visitThreadIdx(call, Util::numeralDimension(calleeN.back())); }
    else if ( calleeN.startswith(prefix + CUDA_BLOCK_ID_REG  )) { visitBlockIdx (call, Util::numeralDimension(calleeN.back())); }
    else if ( calleeN.startswith(prefix + CUDA_BLOCK_DIM_REG )) { visitBlockDim (call, Util::numeralDimension(calleeN.back())); }
    else if ( calleeN.startswith(prefix + CUDA_GRID_DIM_REG  )) { visitGridDim  (call, Util::numeralDimension(calleeN.back())); }
    else {
        errs() << "ERROR: Called unsupported function: " << calleeN << "\n"; //TODO currently no support for custom functions
    }
} 

void CacheLineReuseAnalysis::visitThreadIdx(CallInst *call, unsigned int dimension) {
    MemAccessDescriptor v(dimension, MAX_DIMENSIONS[dimension]);
    OUTs[call->getParent()][call] = std::vector<MemAccessDescriptor>{v};
}

void CacheLineReuseAnalysis::visitBlockIdx(CallInst *call, unsigned int dimension) {
    OUTs[call->getParent()][call] = std::vector<MemAccessDescriptor>{MemAccessDescriptor(0)};
}


void CacheLineReuseAnalysis::visitBlockDim(CallInst *call, unsigned int dimension) {
    int n = 1;
    for (int i = 0; i < dimensions; i++) {
        n *= MAX_DIMENSIONS[i];
    }
    OUTs[call->getParent()][call] = std::vector<MemAccessDescriptor>{MemAccessDescriptor(n)};
}

void CacheLineReuseAnalysis::visitGridDim(CallInst *call, unsigned int dimension) {
    OUTs[call->getParent()][call] = std::vector<MemAccessDescriptor>{MemAccessDescriptor(1)};
}

void CacheLineReuseAnalysis::visitBinaryOp(function<int(int, int)> f, Instruction * inst) {
    std::vector<MemAccessDescriptor> ops1 = getMADs(inst->getOperand(0));
    std::vector<MemAccessDescriptor> ops2 = getMADs(inst->getOperand(1));
    std::vector<MemAccessDescriptor> result;
    for (MemAccessDescriptor op1 : ops1) {
        for (MemAccessDescriptor op2 : ops2) {
            result.push_back(op1.compute(f, op2));
        }
    }
    OUTs[inst->getParent()][inst] = result;
}

void CacheLineReuseAnalysis::visitSExt(Instruction * inst) {
    // forwarding
    OUTs[inst->getParent()][inst] = getMADs(inst->getOperand(0));
}

void CacheLineReuseAnalysis::visitTrunc(Instruction * inst) {
    // forwarding
    OUTs[inst->getParent()][inst] = getMADs(inst->getOperand(0));
}

void CacheLineReuseAnalysis::visitBitCast(Instruction * inst) {
    // forwarding
    OUTs[inst->getParent()][inst] = getMADs(inst->getOperand(0));
}


void CacheLineReuseAnalysis::visitGetElementPtr(Instruction * inst) {
    // forwarding
    OUTs[inst->getParent()][inst] = getMADs(inst->getOperand(1));
}

void CacheLineReuseAnalysis::visitICmp(ICmpInst * inst) {
    switch (inst->getPredicate()) {
        case ICmpInst::Predicate::ICMP_EQ:  visitBinaryOp(std::equal_to<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_UGT: visitBinaryOp(std::greater<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_UGE: visitBinaryOp(std::greater_equal<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_ULT: visitBinaryOp(std::less<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_ULE: visitBinaryOp(std::less_equal<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_SGT: visitBinaryOp(std::greater<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_SGE: visitBinaryOp(std::greater_equal<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_SLT: visitBinaryOp(std::less<int>(), inst); break;
        case ICmpInst::Predicate::ICMP_SLE: visitBinaryOp(std::less_equal<int>(), inst); break;
        default: errs() << "ERROR: Unsupported icmp predicate for instruction " << *inst << "\n";
    }
}

std::vector<MemAccessDescriptor> CacheLineReuseAnalysis::getMADs(Value * v) {
    if (ConstantInt * intval = dyn_cast<ConstantInt>(v)) {
        return std::vector<MemAccessDescriptor>{MemAccessDescriptor(intval->getValue().getSExtValue())};
    } else if (Instruction * inst = dyn_cast<Instruction>(v)) {
        return OUTs[inst->getParent()][inst];
    } else if (isa<Argument>(v)) {
        return std::vector<MemAccessDescriptor>{MemAccessDescriptor(10000)};
    } else if (isa<UndefValue>(v)) {
        return std::vector<MemAccessDescriptor>();
    } else {
        errs() << "ERROR: Unknown operand type from value " << *v << "\n";
        return std::vector<MemAccessDescriptor>();
    }
}


char CacheLineReuseAnalysis::ID = 0;
static RegisterPass<CacheLineReuseAnalysis> X("clr", "Cache Line Re-Use Analysis Pass");
