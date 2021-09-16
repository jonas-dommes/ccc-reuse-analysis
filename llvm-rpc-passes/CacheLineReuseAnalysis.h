#ifndef CACHE_LINE_REUSE_ANALYSIS_H
#define CACHE_LINE_REUSE_ANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/PostDominators.h"

#include <set>
#include <vector>

#include "MemAccessDescriptor.h"
#include "GridAnalysisPass.h"

using namespace llvm;

struct CacheLineReuseInfo {
    bool cacheLineReuse;
    bool dataDependent;
};

struct AnalysisPlanNode {
    BasicBlock * block;
    bool isLoopHeader;
    int loopIteration;
};

class CacheLineReuseAnalysis : public FunctionPass {

  public:
    static char ID;
    
    CacheLineReuseAnalysis() : FunctionPass(ID) {}
    //~CacheLineReuseAnalysis();
    virtual void getAnalysisUsage(AnalysisUsage &au) const;
    virtual bool runOnFunction(Function &F);
    //virtual bool doFinalization(Module &M);
    bool isCacheLineReuse();
    bool isDataDependent();
    bool isErrored();

  private:
    std::string             m_kernelName;
    bool                    m_dynamicMode;
    LoopInfo               *m_loopInfo;
    PostDominatorTree      *m_postDomT;
    DominatorTree          *m_domT;
    GridAnalysisPass       *m_gridAnalysis;
    int                     dimensions;
    bool                    clrErrors;

    std::set<Instruction*>  memops;
    std::set<Instruction*>  relevantInstructions;
    std::set<Loop *>        relevantLoops;
    std::set<PHINode *>     loopPhis;

    std::map<Instruction*, CacheLineReuseInfo>  memopsDiagnostics;

    std::map<BasicBlock*, std::map<Instruction*, std::vector<MemAccessDescriptor>>> INs;
    std::map<BasicBlock*, std::map<Instruction*, std::vector<MemAccessDescriptor>>> OUTs;
    std::vector<AnalysisPlanNode> worklist;
    std::map<StringRef, std::set<int>> accessedCacheLines;
    std::vector<std::string> report;

    void visitBB(BasicBlock *block);
    void visitInst(Instruction *inst);

    void visitCall(CallInst *call);
    void visitThreadIdx(CallInst *call, unsigned int dimension);
    void visitBlockIdx(CallInst *call, unsigned int dimension);
    void visitBlockDim(CallInst *call, unsigned int dimension);
    void visitGridDim(CallInst *call, unsigned int dimension);

    void visitBinaryOp(function<int(int, int)> f, Instruction * inst);
    void visitSExt(Instruction * inst);
    void visitZExt(Instruction * inst);
    void visitTrunc(Instruction * inst);
    void visitICmp(ICmpInst * inst);
    void visitSelect(SelectInst * inst);
    void visitBitCast(Instruction * inst);
    void visitGetElementPtr(Instruction * inst);
    void visitAddrSpaceCast(Instruction * inst);
    void visitPhi(PHINode * inst);
    
    std::vector<MemAccessDescriptor> getMADs(Value * v);

    int getDimensionality();
    void getLoopAnalysisPlan(BasicBlock *header, int loopIteration);
    void preprocess(Function &function, std::set<Instruction*>& memops, std::set<Instruction*>& relevantInstructions);
    void simulateMemoryAccess(Instruction *inst);
    Value * getAccessedSymbolPtr(Value * v);
    StringRef getAccessedSymbolName(Value * v);
    bool isCachedMemoryAccess(Instruction * inst);

};

#endif
