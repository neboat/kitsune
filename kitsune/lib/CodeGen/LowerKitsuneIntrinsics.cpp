//===- LowerKitsuneIntrinsics.cpp - Lower kitsune intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's intrinsics
//
//===----------------------------------------------------------------------===//

#include "kitsune/CodeGen/LowerKitsuneIntrinsics.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

#include <map>
#include <vector>

#define DEBUG_TYPE "kit-lower-intrinsics"

using namespace llvm;

namespace {

using KitsuneRuntimeFuncMap = std::map<Intrinsic::ID, LibFunc>;

/// Kitsune runtime functions for any tapir target.
static const KitsuneRuntimeFuncMap kitFuncs = {
    {Intrinsic::kit_enable_verbose, LibFunc_kitrt_enable_verbose},
};

/// Kitsune runtime functions for the pthreads tapir target.
static const KitsuneRuntimeFuncMap kitPthreadsFuncs = {
    {Intrinsic::kit_finalize, LibFunc_kitpthr_finalize},
    {Intrinsic::kit_initialize, LibFunc_kitpthr_initialize},
    {Intrinsic::kit_sync_threads, LibFunc_kitpthr_sync},
    {Intrinsic::kit_async_launch_threads, LibFunc_kitpthr_launch},
};

/// Kitsune runtime functions for the cuda tapir target.
static const KitsuneRuntimeFuncMap kitCudaFuncs = {
    {Intrinsic::kit_async_launch_kernel, LibFunc_kitcuda_launch_kernel},
    {Intrinsic::kit_async_prefetch_dtoh, LibFunc_kitcuda_prefetch_dtoh},
    {Intrinsic::kit_async_prefetch_htod, LibFunc_kitcuda_prefetch_htod},
    {Intrinsic::kit_enable_refine_launches,
     LibFunc_kitcuda_enable_refine_launches},
    {Intrinsic::kit_finalize, LibFunc_kitcuda_finalize},
    {Intrinsic::kit_initialize, LibFunc_kitcuda_initialize},
    {Intrinsic::kit_mobile_alloc, LibFunc_kitcuda_managed_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_kitcuda_managed_free},
    {Intrinsic::kit_set_fixed_tpb, LibFunc_kitcuda_set_fixed_tpb},
    {Intrinsic::kit_set_max_tpb, LibFunc_kitcuda_set_max_tpb},
    {Intrinsic::kit_symbol_device_ptr, LibFunc_kitcuda_symbol_device_ptr},
    {Intrinsic::kit_symbol_memcpy_dtoh, LibFunc_kitcuda_symbol_memcpy_dtoh},
    {Intrinsic::kit_symbol_memcpy_htod, LibFunc_kitcuda_symbol_memcpy_htod},
    {Intrinsic::kit_sync_stream, LibFunc_kitcuda_sync_stream},
    {Intrinsic::kit_thread_stream, LibFunc_kitcuda_get_thread_stream},
};

/// Kitsune runtime functions for the hip tapir target.
static const KitsuneRuntimeFuncMap kitHipFuncs = {
    {Intrinsic::kit_async_launch_kernel, LibFunc_kithip_launch_kernel},
    {Intrinsic::kit_async_prefetch_dtoh, LibFunc_kithip_prefetch_dtoh},
    {Intrinsic::kit_async_prefetch_htod, LibFunc_kithip_prefetch_htod},
    {Intrinsic::kit_enable_y_axis_launches,
     LibFunc_kithip_enable_y_axis_launches},
    {Intrinsic::kit_finalize, LibFunc_kithip_finalize},
    {Intrinsic::kit_enable_xnack, LibFunc_kithip_enable_xnack},
    {Intrinsic::kit_initialize, LibFunc_kithip_initialize},
    {Intrinsic::kit_mobile_alloc, LibFunc_kithip_managed_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_kithip_managed_free},
    {Intrinsic::kit_set_fixed_tpb, LibFunc_kithip_set_fixed_tpb},
    {Intrinsic::kit_set_max_tpb, LibFunc_kithip_set_max_tpb},
    {Intrinsic::kit_symbol_device_ptr, LibFunc_kithip_symbol_device_ptr},
    {Intrinsic::kit_symbol_memcpy_dtoh, LibFunc_kithip_symbol_memcpy_dtoh},
    {Intrinsic::kit_symbol_memcpy_htod, LibFunc_kithip_symbol_memcpy_htod},
    {Intrinsic::kit_sync_stream, LibFunc_kithip_sync_stream},
    {Intrinsic::kit_thread_stream, LibFunc_kithip_get_thread_stream},
};

/// Runtime library function maps for tapir targets that have a corresponding
/// kitsune runtime.
static const std::map<TTID, KitsuneRuntimeFuncMap> kitTTFuncs = {
    {TTID::Pthreads, kitPthreadsFuncs},
    {TTID::Cuda, kitCudaFuncs},
    {TTID::Hip, kitHipFuncs},
};

/// When lowering the kitsune intrinsics, some arguments may need to be dropped
/// or reordered. The values in this map are the order in which the source
/// operands should appear in the call to the runtime function. For instance,
/// if the value is {2, 1, 3}, it indicates that the first argument to the
/// intrinsic is dropped, while the second and third arguments are swapped.
/// For e.g., if Intrinsic::example were to be lowered to LibFunc_example with
/// the argument map {2, 1, 3}, then this call:
///
///     call void llvm.example(i32 2, ptr writeonly a, ptr readonly b, i64 c)
///
/// would be lowered to
///
///     call void example(ptr readonly b, ptr writeonly a, i64 c)
///
/// Note that the attributes on the call arguments have been preserved. In fact,
/// this is the primary motivation for having this map, since, without it, these
/// attributes would likely be lost.
///
/// NOTE: It is not clear if there is any advantage to preserving these
/// attributes since this pass runs as part of codegen, at which point the
/// optimization pipeline has already run. Even so, it is probably a good idea
/// to preserve these attributes if only to avoid any unnecessary surprises if
/// this part of the code were ever moved elsewhere, or if a (very late)
/// optimization pass were added after this pass.
///
/// If an intrinsic does not contain an entry in this map, a custom lowering
/// function must be provided for it. If the arguments to an intrinsic must be
/// modified before passing them to the runtime function, the lowering *MUST* be
/// handled with a custom lowering function.
static const std::map<Intrinsic::ID, std::vector<unsigned>> kitRTArgMap = {
    {Intrinsic::kit_async_launch_threads, {1, 2, 3, 4, 5}},
    {Intrinsic::kit_sync_threads, {1}},
    {Intrinsic::kit_async_memcpy_dtoh, {1, 2, 3, 4}},
    {Intrinsic::kit_async_memcpy_htod, {1, 2, 3, 4}},
    {Intrinsic::kit_async_prefetch_dtoh, {1, 3}},
    {Intrinsic::kit_async_prefetch_htod, {1, 3}},
    {Intrinsic::kit_enable_refine_launches, {1}},
    {Intrinsic::kit_enable_verbose, {}},
    {Intrinsic::kit_enable_xnack, {}},
    {Intrinsic::kit_enable_y_axis_launches, {}},
    {Intrinsic::kit_finalize, {}},
    {Intrinsic::kit_initialize, {}},
    {Intrinsic::kit_memcpy_dtoh, {1, 2, 3}},
    {Intrinsic::kit_memcpy_htod, {1, 2, 3}},
    {Intrinsic::kit_set_fixed_tpb, {1}},
    {Intrinsic::kit_set_max_tpb, {1}},
    {Intrinsic::kit_symbol_device_ptr, {1, 2}},
    {Intrinsic::kit_symbol_memcpy_dtoh, {2, 1, 3}},
    {Intrinsic::kit_symbol_memcpy_htod, {2, 1, 3}},
    {Intrinsic::kit_sync_stream, {1}},
    {Intrinsic::kit_thread_stream, {}},
};

/// Main implementation class to lower Kitsune intrinsics.
class LowerKitsuneIntrinsics {
private:
  const TapirTargetInfo &tgi;
  TargetLibraryInfo &tli;

private:
  /// Some runtime intrinsics take the tapir target id as the first argument.
  /// Get the tapir target from this argument. It is an error to call this
  /// function with a call that is not a kitsune runtime intrinsic and does not
  /// have a valid tapir target as the first argument.
  TTID getTTID(CallInst &call) const {
    if (auto *cint = dyn_cast<ConstantInt>(call.getArgOperand(0)))
      if (std::optional<TTID> tt = createTTIDFrom(cint->getZExtValue()))
        return *tt;
    llvm_unreachable("getTTID: Not a valid tapir target id");
  }

  FunctionCallee getOrInsertLibFunc(Module &m, LibFunc libFunc) {
    FunctionCallee f = llvm::getOrInsertLibFunc(&m, tli, libFunc);
    inferNonMandatoryLibFuncAttrs(*cast<Function>(f.getCallee()), tli);
    return f;
  }

  FunctionCallee getOrInsertLibFunc(Module &m, Intrinsic::ID id) {
    assert(kitFuncs.find(id) != kitFuncs.end() &&
           "getRuntimeFunc: No kitsune library function for intrinsic");
    return getOrInsertLibFunc(m, kitFuncs.at(id));
  }

  FunctionCallee getOrInsertLibFunc(Module &m, TTID tt, Intrinsic::ID id) {
    assert(kitTTFuncs.find(tt) != kitTTFuncs.end() &&
           "getRuntimeFunc: Invalid tapir target for intrinsic");

    const KitsuneRuntimeFuncMap &funcs = kitTTFuncs.at(tt);
    assert(funcs.find(id) != funcs.end() &&
           "getRuntimeFunc: No kitsune library function for tapir "
           "target");

    return getOrInsertLibFunc(m, funcs.at(id));
  }

  FunctionCallee getMobileMemFunc(Module &m, Intrinsic::ID id, LibFunc SerialLibFunc) {
    /// TODO: Currently, this is very naive and simply looks at the primary
    /// target. This will not work correctly in multi-target mode. But that
    /// requires a more sophisticated analysis which should be implemented
    /// eventually.
    std::optional<TTID> tt = tgi.getTTIDOrNull();
    if (not tt)
      return getOrInsertLibFunc(m, SerialLibFunc);

    switch (*tt) {
    case TTID::Cuda:
      return getOrInsertLibFunc(m, TTID::Cuda, id);
    case TTID::Hip:
      return getOrInsertLibFunc(m, TTID::Hip, id);
    case TTID::Custom:
      // TODO: A custom tapir target may require a custom memory allocator.
      // Currently, there is no way to have the plugin specify a memory
      // allocator to use, so just default to using libc's malloc.
    case TTID::Nolo:
      // When using the 'nolo' tapir target, we should never get here, but in
      // case we do, just default to using libc's malloc.
    case TTID::OpenCilk:
    case TTID::Pthreads:
    case TTID::Serial:
      return getOrInsertLibFunc(m, SerialLibFunc);
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::OpenMP:
    case TTID::Qthreads:
    case TTID::Realm:
      // These tapir targets are not fully supported yet, but add them to this
      // switch to ensure that a warning is emitted when a new tapir target is
      // added.
      break;
    }
    llvm_unreachable("getMemAllocFunc: TTID not handled");
  }

  /// Get the kitsune runtime function that will replace the intrinsic called in
  /// the given call instruction.
  FunctionCallee getRuntimeFunc(CallInst &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    Module &m = *call.getModule();

    switch (id) {
    case Intrinsic::kit_mobile_alloc:
      return getMobileMemFunc(m, id, LibFunc_malloc);

    case Intrinsic::kit_mobile_free:
      return getMobileMemFunc(m, id, LibFunc_free);

    case Intrinsic::kit_enable_verbose:
      // Intrinsics with runtime functions that are independent of a tapir
      // target.
      return getOrInsertLibFunc(m, id);

    case Intrinsic::kit_enable_xnack:
      // Intrinsics that are exclusive to the hip tapir target
      return getOrInsertLibFunc(m, TTID::Hip, id);

    default:
      // Intrinsics with runtime functions dependent on the tapir target.
      return getOrInsertLibFunc(m, getTTID(call), id);
    }
  }

  /// Return a new attribute list which is exactly the same as the given
  /// attribute list \ref attrs except that the attributes at index \ref src of
  /// \ref call's attribute list are added to index \ref dst of \ref attrs. The
  /// newly created attribute list is returned.
  AttributeList addAttrsFrom(AttributeList attrs, unsigned dst,
                             const CallInst &call, unsigned src) {
    LLVMContext &ctx = call.getContext();
    AttributeList callAttrs = call.getAttributes();
    for (const Attribute &attr : callAttrs.getAttributes(src))
      attrs = attrs.addAttributeAtIndex(ctx, dst, attr);
    return attrs;
  }

  /// Return a new attribute list which is exactly the same as the given
  /// attribute list \ref attrs except that the attributes at index \ref src of
  /// \ref call's attribute list are added to index \ref src of \ref attrs. The
  /// newly created attribute list is returned.
  AttributeList addAttrsFrom(AttributeList attrs, const CallInst &call,
                             unsigned src) {
    return addAttrsFrom(attrs, src, call, src);
  }

  /// Create a new attribute list that will eventually be applied to the
  /// replacement of the given call instruction. If \ref argMap[i] = j, the
  /// attributes from the j'th argument of \ref call will be added to index i
  /// of the new attribute list that is returned.
  AttributeList createNewAttrList(const CallInst &call,
                                  ArrayRef<unsigned> argMap) {
    AttributeList attrs;
    attrs = addAttrsFrom(attrs, call, AttributeList::FunctionIndex);
    attrs = addAttrsFrom(attrs, call, AttributeList::ReturnIndex);
    for (size_t i = 0; i < argMap.size(); ++i) {
      unsigned isrc = AttributeList::FirstArgIndex + argMap[i];
      unsigned idst = AttributeList::FirstArgIndex + i;
      attrs = addAttrsFrom(attrs, idst, call, isrc);
    }
    return attrs;
  }

  /// Create a new call to the given function to replace an existing call. The
  /// debug info, metadata, calling convention and tail call kind will be copied
  /// over from the original call. However, the attributes will not be copied.
  /// The new call is returned, but the original call will remain unchanged.
  ///
  /// The attributes are not copied because there are some intrinsics where the
  /// attributes cannot be copied over directly. To avoid having conditional
  /// statements in this function, we require the attributes to be copied over
  /// by callers.
  CallInst *createNewCallFor(CallInst &call, FunctionCallee f,
                             ArrayRef<Value *> args) {
    StringRef name = call.getName();
    BasicBlock::iterator pos = call.getIterator();

    CallInst *newCall = CallInst::Create(f, args, name, pos);
    newCall->cloneDebugInfoFrom(&call);
    newCall->copyMetadata(call);
    newCall->setCallingConv(call.getCallingConv());

    // Because the result of the lowered intrinsic must be cast to a different
    // "type", tail calls cannot be guaranteed.
    CallInst::TailCallKind tck = call.getTailCallKind();
    if (tck == CallInst::TCK_MustTail)
      newCall->setTailCallKind(CallInst::TCK_Tail);
    else
      newCall->setTailCallKind(tck);

    return newCall;
  }

  /// Depending on the tapir target to be used, this may not replace the call.
  /// In that case, return false. Otherwise, return true.
  bool lowerMobileAlloc(CallInst &call) {
    FunctionCallee f = getRuntimeFunc(call);
    if (not f.getCallee())
      return false;

    std::vector<Value *> args;
    for (Use &arg : call.args())
      args.push_back(arg.get());
    BasicBlock::iterator pos = call.getIterator();
    Type *retTy = call.getType();

    // The result of the new call will be a pointer in the default address
    // space. However, all uses of the call will be in the mobile address
    // space.
    CallInst *newCall = createNewCallFor(call, f, args);
    CastInst *cst =
        CastInst::Create(Instruction::AddrSpaceCast, newCall, retTy, "", pos);
    newCall->setAttributes(createNewAttrList(call, {0}));

    cst->moveAfter(newCall);
    call.replaceAllUsesWith(cst);
    call.eraseFromParent();

    return true;
  }

  /// Depending on the tapir target to be used, this may not replace the call.
  /// In the case, return false, otherwise, return true.
  bool lowerMobileFree(CallInst &call) {
    FunctionCallee f = getRuntimeFunc(call);
    if (not f.getCallee())
      return false;

    LLVMContext &ctx = call.getContext();
    Type *ptrTy = PointerType::getUnqual(ctx);
    BasicBlock::iterator pos = call.getIterator();

    // The call expects a pointer in the default address space, but the
    // argument to Kitsune's intrinsic will be in the mobile address space.
    CastInst *cst = CastInst::Create(Instruction::AddrSpaceCast,
                                     call.getArgOperand(0), ptrTy, "", pos);
    CallInst *newCall = createNewCallFor(call, f, cst);
    newCall->setAttributes(createNewAttrList(call, {0}));

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// Lower the kernel launch intrinsic. This is a vararg intrinsic, but the
  /// corresponding runtime functions need the arguments to be passed an array
  /// of pointers to the arguments. We implement this by creating a stack slot
  /// for each argument, and an array of pointers, each of which is a pointer to
  /// one of these stack slots. The runtime function is passed a pointer to
  /// this array of pointers.
  bool lowerLaunchKernel(CallInst &call) {
    LLVMContext &ctx = call.getContext();
    Type *i64 = Type::getInt64Ty(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Constant *c0 = ConstantInt::get(i64, 0);

    BasicBlock &bbEntry = call.getParent()->getParent()->getEntryBlock();

    std::vector<Value *> kernelArgs = getKernelArgumentsFromLaunch(call);
    ArrayType *arrTy = ArrayType::get(ptrTy, kernelArgs.size());
    AllocaInst *argArray = new AllocaInst(arrTy, 0, "", bbEntry.begin());
    for (size_t i = 0; i < kernelArgs.size(); ++i) {
      Constant *ci = ConstantInt::get(i64, i);
      Value *indices[] = {c0, ci};
      Value *kernelArg = kernelArgs[i];
      Type *argTy = kernelArg->getType();

      AllocaInst *slot = new AllocaInst(argTy, 0, "", argArray->getIterator());
      (void)new StoreInst(kernelArg, slot, call.getIterator());
      GetElementPtrInst *argOffset = GetElementPtrInst::CreateInBounds(
          arrTy, argArray, indices, "", call.getIterator());
      (void)new StoreInst(slot, argOffset, call.getIterator());
    }

    // The attributes can mostly be copied over to the new call. However, the
    // 3rd argument in the runtime call will be the packed argument array, so
    // those attributes cannot be copied over from the intrinsic call.
    unsigned attr0 = AttributeList::FirstArgIndex;
    AttributeList attrs;
    attrs = addAttrsFrom(attrs, call, AttributeList::FunctionIndex);
    attrs = addAttrsFrom(attrs, call, AttributeList::ReturnIndex);
    attrs = addAttrsFrom(attrs, attr0 + 0, call, attr0 + 1);
    attrs = addAttrsFrom(attrs, attr0 + 1, call, attr0 + 2);
    attrs = addAttrsFrom(attrs, attr0 + 3, call, attr0 + 3);
    attrs = addAttrsFrom(attrs, attr0 + 4, call, attr0 + 4);
    attrs = addAttrsFrom(attrs, attr0 + 5, call, attr0 + 5);
    attrs = addAttrsFrom(attrs, attr0 + 6, call, attr0 + 6);
    attrs = attrs.addAttributeAtIndex(ctx, attr0 + 2, Attribute::NonNull);

    FunctionCallee rtFunc = getRuntimeFunc(call);
    Value *args[] = {
        call.getArgOperand(1), call.getArgOperand(2), argArray,
        call.getArgOperand(3), call.getArgOperand(4), call.getArgOperand(5),
        call.getArgOperand(6),
    };
    CallInst *newCall = createNewCallFor(call, rtFunc, args);

    newCall->setAttributes(attrs);
    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// Replace the Kitsune intrinsic called in the given instruction with an
  /// appropriate runtime function to be called with the given arguments.
  /// Always returns true.
  bool lowerIntrinsicDefault(CallInst &call) {
    assert((kitRTArgMap.find(call.getIntrinsicID()) != kitRTArgMap.end()) &&
           "Intrinsic supports default lowering");

    ArrayRef argMap = kitRTArgMap.at(call.getIntrinsicID());
    FunctionCallee f = getRuntimeFunc(call);
    std::vector<Value *> args;
    for (unsigned argNo : argMap)
      args.push_back(call.getArgOperand(argNo));

    CallInst *newCall = createNewCallFor(call, f, args);
    newCall->setAttributes(createNewAttrList(call, argMap));

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// The given call instruction is a call to a kitsune intrinsic. This may
  /// lower it (in some cases, the instruction will not be lowered - for
  /// instance if the primary tapir target is one that does not permit
  /// lowering). Returns true if the call to the instrinsic was replaced, false
  /// otherwise.
  bool lowerIntrinsic(CallInst &call) {
    bool changed = false;

    switch (call.getIntrinsicID()) {
    case Intrinsic::kit_mobile_alloc:
      changed |= lowerMobileAlloc(call);
      break;

    case Intrinsic::kit_mobile_free:
      changed |= lowerMobileFree(call);
      break;

    case Intrinsic::kit_async_launch_kernel:
      changed |= lowerLaunchKernel(call);
      break;

    case Intrinsic::kit_enable_verbose:
    case Intrinsic::kit_enable_xnack: {
      // The only argument is an immediate flag. If the flag is false, the
      // corresponding runtime function should not be called.
      if (cast<ConstantInt>(call.getArgOperand(0))->isZero())
        call.eraseFromParent();
      else
        lowerIntrinsicDefault(call);
      changed |= true;
      break;
    }

    case Intrinsic::kit_enable_y_axis_launches: {
      // The first argument is the tapir target id. The second is a boolean
      // immediate flag. If the flag is false, the corresponding runtime
      // function should not be called.
      if (cast<ConstantInt>(call.getArgOperand(1))->isZero())
        call.eraseFromParent();
      else
        lowerIntrinsicDefault(call);
      changed |= true;
      break;
    }

    default:
      changed |= lowerIntrinsicDefault(call);
      break;
    }

    return changed;
  }

public:
  LowerKitsuneIntrinsics(const TapirTargetInfo &tgi, TargetLibraryInfo &tli)
      : tgi(tgi), tli(tli) {}

  bool run(Function &f) {
    bool changed = false;
    std::optional<TTID> tt = tgi.getTTIDOrNull();
    if (not tt or *tt == TTID::Nolo)
      return changed;

    std::vector<CallInst *> calls;
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      if (auto *call = dyn_cast<CallInst>(&*i)) {
        if (isKitsuneIntrinsic(call->getIntrinsicID()))
          calls.push_back(call);
      } else if (auto *invoke = dyn_cast<InvokeInst>(&*i)) {
        if (isKitsuneIntrinsic(invoke->getIntrinsicID()))
          llvm_unreachable("Invoke of kitsune intrinsic");
      }
    }

    for (CallInst *call : calls)
      changed |= lowerIntrinsic(*call);

    return changed;
  }
};

/// Legacy pass to compile the embedded bitcode to fat binaries.
class LowerKitsuneIntrinsicsLegacyPass : public ModulePass {
public:
  LowerKitsuneIntrinsicsLegacyPass() : ModulePass(ID) {
    initializeLowerKitsuneIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Lower Kitsune intrinsics"; }

  void getAnalysisUsage(AnalysisUsage &au) const override {
    au.addRequired<TapirTargetAnalysisWrapperPass>();
    au.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnModule(Module &m) override {
    const TapirTargetInfo &tgi =
        getAnalysis<TapirTargetAnalysisWrapperPass>().getResult();

    bool changed = false;
    for (Function &f : m) {
      TargetLibraryInfo &tli =
          getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(f);

      changed |= LowerKitsuneIntrinsics(tgi, tli).run(f);
    }

    return changed;
  }

public:
  static char ID;
};

} // namespace

char LowerKitsuneIntrinsicsLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(LowerKitsuneIntrinsicsLegacyPass, DEBUG_TYPE,
                      "Lower Kitsune intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TapirTargetAnalysisWrapperPass)
INITIALIZE_PASS_END(LowerKitsuneIntrinsicsLegacyPass, DEBUG_TYPE,
                    "Lower Kitsune intrinsics", false, false)

ModulePass *llvm::createLowerKitsuneIntrinsicsLegacyPass() {
  return new LowerKitsuneIntrinsicsLegacyPass();
}

PreservedAnalyses LowerKitsuneIntrinsicsPass::run(Module &m,
                                                  ModuleAnalysisManager &mam) {
  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);

  bool changed = false;
  for (Function &f : m) {
    TargetLibraryInfo &tli = fam.getResult<TargetLibraryAnalysis>(f);

    changed |= LowerKitsuneIntrinsics(tgi, tli).run(f);
  }

  // If any kitsune intrinsics were replaced, the call graph will have changed,
  // but other analyses will not have been invalidated.
  if (changed) {
    PreservedAnalyses pa;
    pa.preserve<FunctionAnalysisManagerCGSCCProxy>();
    pa.preserveSet<AllAnalysesOn<Function>>();
    return pa;
  } else {
    return PreservedAnalyses::all();
  }
}
