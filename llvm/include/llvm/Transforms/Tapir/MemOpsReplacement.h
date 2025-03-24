#ifndef LLVM_TRANSFORMS_TAPIR_MEMOPS_REPLACEMENT_H
#define LLVM_TRANSFORMS_TAPIR_MEMOPS_REPLACEMENT_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class MemOpsReplacementPass : public PassInfoMixin<MemOpsReplacementPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_MEMOPS_REPLACEMENT_H
