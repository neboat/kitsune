#include "llvm/Transforms/Tapir/MemOpsReplacement.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "memops-replacement"

namespace {

void collectMemInsns(Function &F, SmallVector<IntrinsicInst *> &MemInsns,
                     Intrinsic::ID ID, const char *Name) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == ID) {
          Value *IsVolatile = II->getArgOperand(3);
          if (ConstantInt *CI = dyn_cast<ConstantInt>(IsVolatile)) {
            if (CI->isZero()) {
              MemInsns.push_back(II);
              LLVM_DEBUG(dbgs()
                         << "Identified " << Name << ": " << *II << "\n");
            }
          }
        }
      }
    }
  }
}

void replaceMemInsnsWithCalls(LLVMContext &Ctx,
                              SmallVector<IntrinsicInst *> &Insts, Function *Fn,
                              const char *Name) {
  for (IntrinsicInst *Inst : Insts) {
    Value *Dest = Inst->getArgOperand(0);
    Value *Src = Inst->getArgOperand(1);
    Value *Count = Inst->getArgOperand(2);
    // If Count is 32-bit, we need to cast it to 64-bit
    if (Count->getType() == Type::getInt32Ty(Ctx)) {
      Count = new ZExtInst(Count, Type::getInt64Ty(Ctx), "", Inst);
    }
    CallInst *Call = CallInst::Create(Fn, {Dest, Src, Count});
    LLVM_DEBUG(dbgs() << "Inserting " << Name << " call: " << *Call << " to "
                      << "replace " << *Inst << "\n");
    Call->insertBefore(Inst);
    // If MemCpyInst was a tail call, let MemCpyCall be a tail call as well
    if (Inst->isTailCall()) {
      Call->setTailCall();
    }
    Inst->replaceAllUsesWith(Call);
    Inst->eraseFromParent();
  }
}

void replaceMemCpy(Function &F) {
  // Find all llvm.memcpy intrinsics in the function
  SmallVector<IntrinsicInst *> MemCpyInsts;
  collectMemInsns(F, MemCpyInsts, Intrinsic::memcpy, "memcpy");
  if (MemCpyInsts.empty()) {
    return;
  }
  Module &M = *F.getParent();
  // Insert declaration for __kitcuda_memcpy into the module
  // void __kitcuda_memcpy(void *dest, const void *src, size_t count)
  FunctionCallee MemCpyFnCallee = M.getOrInsertFunction(
      "__kitcuda_memcpy", Type::getVoidTy(F.getContext()),
      PointerType::getUnqual(F.getContext()),
      PointerType::getUnqual(F.getContext()), Type::getInt64Ty(F.getContext()));
  Function *MemCpyFn = cast<Function>(MemCpyFnCallee.getCallee());
  {
    MemCpyFn->addFnAttr(Attribute::NoUnwind);
    MemCpyFn->addFnAttr(Attribute::WillReturn);
    MemCpyFn->addParamAttr(0, Attribute::WriteOnly);
    MemCpyFn->addParamAttr(0, Attribute::NoCapture);
    MemCpyFn->addParamAttr(1, Attribute::ReadOnly);
    MemCpyFn->addParamAttr(1, Attribute::NoCapture);
    MemCpyFn->setCallingConv(CallingConv::C);
    MemCpyFn->setLinkage(GlobalValue::ExternalLinkage);
  }
  replaceMemInsnsWithCalls(F.getContext(), MemCpyInsts, MemCpyFn, "memcpy");
}

void replaceMemMove(Function &F) {
  SmallVector<IntrinsicInst *> MemMoveInsts;
  collectMemInsns(F, MemMoveInsts, Intrinsic::memmove, "memmove");
  if (MemMoveInsts.empty()) {
    return;
  }

  Module &M = *F.getParent();
  // Insert declaration for __kitcuda_memmove into the module
  // void __kitcuda_memmove(void *dest, const void *src, size_t count)
  FunctionCallee MemMoveFnCallee = M.getOrInsertFunction(
      "__kitcuda_memmove", Type::getVoidTy(F.getContext()),
      PointerType::getUnqual(F.getContext()),
      PointerType::getUnqual(F.getContext()), Type::getInt64Ty(F.getContext()));
  Function *MemMoveFn = cast<Function>(MemMoveFnCallee.getCallee());
  {
    MemMoveFn->addFnAttr(Attribute::NoUnwind);
    MemMoveFn->addFnAttr(Attribute::WillReturn);
    MemMoveFn->addParamAttr(0, Attribute::NoCapture);
    MemMoveFn->addParamAttr(1, Attribute::NoCapture);
    MemMoveFn->setCallingConv(CallingConv::C);
    MemMoveFn->setLinkage(GlobalValue::ExternalLinkage);
  }
  replaceMemInsnsWithCalls(F.getContext(), MemMoveInsts, MemMoveFn, "memmove");
}

void replaceMemSet(Function &F) {
  // Find all llvm.memset intrinsics in the function
  SmallVector<IntrinsicInst *> MemSetInsts;
  collectMemInsns(F, MemSetInsts, Intrinsic::memset, "memset");
  if (MemSetInsts.empty()) {
    return;
  }
  Module &M = *F.getParent();
  // Insert declaration for __kitcuda_memset into the module
  // void __kitcuda_memset(void *dest, int val, size_t count)
  FunctionCallee MemSetFnCallee = M.getOrInsertFunction(
      "__kitcuda_memset", Type::getVoidTy(F.getContext()),
      PointerType::getUnqual(F.getContext()), Type::getInt8Ty(F.getContext()),
      Type::getInt64Ty(F.getContext()));
  Function *MemSetFn = cast<Function>(MemSetFnCallee.getCallee());
  {
    MemSetFn->addFnAttr(Attribute::NoUnwind);
    MemSetFn->addFnAttr(Attribute::WillReturn);
    MemSetFn->addParamAttr(0, Attribute::WriteOnly);
    MemSetFn->addParamAttr(0, Attribute::NoCapture);
    MemSetFn->setCallingConv(CallingConv::C);
    MemSetFn->setLinkage(GlobalValue::ExternalLinkage);
  }
  replaceMemInsnsWithCalls(F.getContext(), MemSetInsts, MemSetFn, "memset");
}

} // namespace

PreservedAnalyses MemOpsReplacementPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  replaceMemCpy(F);
  replaceMemMove(F);
  replaceMemSet(F);
  return PreservedAnalyses::none();
}
