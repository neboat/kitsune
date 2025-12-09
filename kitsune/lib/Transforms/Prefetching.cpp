//===- Prefetching.cpp - Generate dtoh/htod prefetch calls ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate prefetch calls to initiate movement of data between host and device.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/Prefetching.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "kit-prefetch"

using namespace llvm;

namespace {

/// Implementation of the prefetch pass. This is currently very simple and
/// inserts device-to-host prefetches immediately before kernel launches as long
/// as prefetching has not been disabled in the tapir target options. This
/// implementation does not distinguish between launches on NVIDIA or AMD GPU's.
/// This is unlikely to be profitable in the general case. This pass will almost
/// certainly have to be split into special prefetching implementations for
/// NVIDIA and AMD and will probably need to query the embedded device modules
/// to determine exactly which GPU to generate prefetches for do something
/// different for each of them. If that happens, this comment should be updated
/// to reflect any changes.
class Prefetch {
private:
  const TTOptions &tto;

private:
  /// Insert prefetches from host to device before the given kernel launch call.
  /// Currently, this inserts the prefetches immediately before the call. It may
  /// be beneficial to insert these earlier in the function. That optimization
  /// may be implemented at some point.
  bool insertPrefetchesHToD(CallBase &call) {
    bool changed = false;

    LLVMContext &ctx = call.getContext();
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *i64Ty = Type::getInt64Ty(ctx);

    Module *m = call.getModule();
    Value *stream = getStreamFromLaunch(call);
    for (Value *arg : getKernelArgumentsFromLaunch(call)) {
      // Don't prefetch hyper-intrinsic arguments.
      if (CallInst *argCall = dyn_cast<CallInst>(arg))
        if (isHyperIntrinsic(argCall->getIntrinsicID()))
          continue;

      if (auto *pty = dyn_cast<PointerType>(arg->getType())) {

        // These should be required to be pointers in Kitsune's mobile address
        // space. Currently, the frontend support for this is a bit spotty. So
        // the Kitsune intrinsics operate on pointers in the default address
        // space. But this really should change at some point.
        //
        // For now, if the pointer is not in the default address space, make
        // sure that it is.
        if (pty->getAddressSpace() != 0)
          arg = CastInst::Create(CastInst::AddrSpaceCast, arg, ptrTy, "",
                                 call.getIterator());

        // The pointer to the data to be prefetched must point to UVM allocated
        // memory. By setting the number of bytes to be prefetched to -1, we are
        // instructing the runtime to prefetch the entire UVM-allocated buffer.
        // The runtime keeps track of this.
        //
        // TODO: Do some analysis to only prefetch the number of bytes that are
        // actually used (or likely to be used) by the kernel.
        Constant *bytes = ConstantInt::get(i64Ty, -1);

        Value *ctt = call.getArgOperand(0);
        FunctionCallee prefetch = Intrinsic::getOrInsertDeclaration(
            m, Intrinsic::kit_async_prefetch_htod);
        (void)CallInst::Create(prefetch, {ctt, arg, bytes, stream}, "",
                               call.getIterator());
      }
    }

    return changed;
  }

  /// Insert prefetches from device to host after the given kernel launch call.
  /// This is not currently implemented.
  bool insertPrefetchesDToH(CallBase &call) {
    // It is not yet clear if this is beneficial, but this placeholder is
    // retained here in case we determine that it is.
    return false;
  }

  /// Insert prefetches for the given kernel launch call if necessary. Return
  /// true if any prefetch calls were inserted, false otherwise.
  bool insertPrefetches(CallBase &call) {
    bool changed = false;

    changed |= insertPrefetchesHToD(call);
    changed |= insertPrefetchesDToH(call);

    return changed;
  }

public:
  Prefetch(const TTOptions &tto) : tto(tto) {}

  /// Iterate over the functions in the module and insert prefetch calls into
  /// them if required. Return false if no prefetch calls were inserted, true
  /// otherwise.
  bool run(Module &m) {
    bool changed = false;
    if (not tto.getGPUPrefetch())
      return changed;

    if (Function *launch = Intrinsic::getDeclarationIfExists(
            &m, Intrinsic::kit_async_launch_kernel))
      for (Use &u : launch->uses())
        if (auto *call = dyn_cast<CallBase>(u.getUser()))
          if (call->getCalledFunction() == launch)
            changed |= insertPrefetches(*call);

    return changed;
  }
};

} // namespace

PreservedAnalyses PrefetchingPass::run(Module &m, ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, there will be nothing to do, so
  // bail out immediately.
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasTTID())
    return PreservedAnalyses::all();

  const TTOptions &tto = tgi.getOptions();

  if (Prefetch(tto).run(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
