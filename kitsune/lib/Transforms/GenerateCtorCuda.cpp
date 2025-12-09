//===- GenerateCtorCuda.cpp - Generate ctor for Kitsune's cuda runtime ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate a global constructor and destructor for Kitsune's cuda runtime
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsImpl.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {

/// Helper class to generate a ctor for kitcuda (Kitsune's runtime for the cuda
/// tapir target). This will also create variables for the fat binary and the
/// bundle containing the fat binary that is needed by the cuda runtime.
///
/// Registering the fat binary image (and all the associated components) is
/// an undocumented portion of the CUDA API. One place to peek for some details
/// hides in the cuda header files; specifially fatbinary_section.h. This shows
/// the following struct that we need to have in the host side code.
///
///    struct fatbinC_Wrapper_t {
///      int magic;
///      int version;
///      const unsigned long long *data;
///      void *filename_or_fatbins;
///    };
///
/// * Per the header, the magic number is 0x466243B1
/// * FATBINC_VERSION is 1 and FATBINC_LINK_VERSION is 2 (more below)
/// * Then section and segments are needed that contains the "fatbin control
///   structure".  This loosely looks like:
///
///        Control section name: ".nvFatBinSegment"
///        Fatbinary section name: ".nv_fatbin"
///        Pre-linked relocatable section: "__nv_relfatbin"
///
/// * The last struct member varies between versions. In the case of version 1
///   it can be a offline filename and for version 2 it is an array of
///   pre-linked fatbins.
///
class GenerateCtorCuda {
private:
  static constexpr int magic = 0x466243B1;
  static constexpr int version = 1;
  static constexpr const char *controlSectionName = ".nvFatBinSegment";

private:
  detail::GetTLI getTLI;
  const TTOptions &tto;
  const detail::GenerateCtorOptions &genCtorOpts;

private:
  /// Create a global variable containing the fat binary "bundle". This
  /// consists of the fat binary and some metadata.
  GlobalVariable *createBundleGV(Module &m, GlobalVariable *fatBin) {
    const DataLayout &dl = m.getDataLayout();

    LLVMContext &ctx = m.getContext();
    Type *i32Ty = Type::getInt32Ty(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *idxTy = dl.getIndexType(ptrTy);
    StructType *bundleTy = StructType::get(/*magic*/ i32Ty,
                                           /*version*/ i32Ty,
                                           /*fat binary data*/ ptrTy,
                                           /*unused*/ ptrTy);

    Constant *zero = ConstantInt::get(idxTy, 0);
    Constant *zeros[] = {zero, zero};

    // Wrap the fatbinary in struct that the CUDA runtime and tools expect.
    Constant *bundleInit = ConstantStruct::get(
        bundleTy, ConstantInt::get(i32Ty, magic),
        ConstantInt::get(i32Ty, version),
        ConstantExpr::getGetElementPtr(fatBin->getValueType(), fatBin, zeros),
        ConstantPointerNull::get(ptrTy));

    GlobalVariable *g = new GlobalVariable(m, bundleTy, /*isConstant*/ true,
                                           GlobalValue::InternalLinkage,
                                           bundleInit, ".kitcuda.bundle");
    g->setSection(controlSectionName);
    g->setAlignment(dl.getPrefTypeAlign(g->getType()));

    return g;
  }

  /// Create a global variable that will contain the "handle" to the fat binary.
  /// The handle is the value returned by __cudaRegisterFatBinary(). The handle
  /// is saved into this global and read from there by the global dtor and
  /// passed to __cudaUnregisterFatBinary().
  GlobalVariable *createBundleHandleGV(Module &m) {
    const DataLayout &dl = m.getDataLayout();

    LLVMContext &ctx = m.getContext();
    PointerType *ptrTy = PointerType::getUnqual(ctx);

    GlobalVariable *g = new GlobalVariable(
        m, ptrTy, /*isConstant*/ false, GlobalValue::InternalLinkage,
        ConstantPointerNull::get(ptrTy), ".kitcuda.handle");
    g->setAlignment(dl.getPointerABIAlignment(0));
    g->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

    return g;
  }

  Function *createCtor(Module &m, const Module &devM, Function *dtor,
                       GlobalVariable *gBundle, GlobalVariable *gBundleHandle) {
    const DataLayout &dl = m.getDataLayout();
    Align alignPtr = dl.getPointerABIAlignment(0);

    LLVMContext &ctx = m.getContext();

    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *i32Ty = Type::getInt32Ty(ctx);
    Type *boolTy = Type::getInt8Ty(ctx);

    ConstantInt *constTT = createConstInt(TTID::Cuda, ctx);
    Constant *czero = ConstantInt::get(i32Ty, 0);

    FunctionType *ctorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *ctor = Function::Create(ctorTy, GlobalValue::InternalLinkage,
                                      ".kitcuda.ctor", &m);

    TargetLibraryInfo &tli = getTLI(*ctor);
    Type *sizeTTy = tli.getSizeTType(m);

    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", ctor));

    FunctionCallee kitrtInitialize =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_initialize);
    builder.CreateCall(kitrtInitialize, {constTT});

    // Enable verbose mode early in the constructor so all verbose statements
    // are printed after the runtime has been initialized.
    FunctionCallee kitrtEnableVerbose =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_enable_verbose);
    builder.CreateCall(
        kitrtEnableVerbose,
        {ConstantInt::get(boolTy, tto.getKitrtVerbose(), false)});

    if (unsigned fixedTPB = tto.getFixedThreadsPerBlock()) {
      FunctionCallee kitrtSetFixedTPB =
          Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_set_fixed_tpb);
      builder.CreateCall(kitrtSetFixedTPB,
                         {constTT, ConstantInt::get(i32Ty, fixedTPB)});
    }

    // If the MaxThreadsPerBlock has not been set, use a value of 1024 anyway.
    // At the time of writing, exceeding this value degrades performance. This
    // might change, and we may even have to set a different value depending
    // on the specific GPU architecture.
    //
    // FIXME: Don't hardcode this value here. Maybe move it to a named constant.
    FunctionCallee kitrtSetMaxTPB =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_set_max_tpb);
    unsigned maxTPB = tto.getMaxThreadsPerBlock();
    if (!maxTPB)
      maxTPB = 1024;
    builder.CreateCall(kitrtSetMaxTPB,
                       {constTT, ConstantInt::get(i32Ty, maxTPB)});

    FunctionCallee kitrtEnableRefineLaunches =
        Intrinsic::getOrInsertDeclaration(
            &m, Intrinsic::kit_enable_refine_launches);
    builder.CreateCall(
        kitrtEnableRefineLaunches,
        {constTT, ConstantInt::get(boolTy, genCtorOpts.refineLaunches)});

    FunctionCallee cudaRegisterFatBinary =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_register_fat_binary);
    Value *bundleHandle = builder.CreateCall(cudaRegisterFatBinary, gBundle);
    builder.CreateAlignedStore(bundleHandle, gBundleHandle, alignPtr);

    // Register any non-constant global variables used in the kernel module.
    // Each of these should have a corresponding global in the host.
    FunctionCallee cudaRegisterVar =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_register_var);
    for (const GlobalVariable &devG : devM.globals()) {
      if (devG.isConstant())
        continue;

      GlobalVariable *hostG = m.getGlobalVariable(devG.getName());
      if (not hostG)
        // This global variable might be local to the kernel module, such as for
        // an array of device-side shared memory.
        continue;

      uint64_t size = dl.getTypeAllocSize(hostG->getType());

      GlobalVariable *gName = createConstString(hostG->getName(), m);
      Constant *gSize = ConstantInt::get(sizeTTy, size);
      Constant *gConst = ConstantInt::get(i32Ty, hostG->isConstant());
      // FIXME?: Why is this always set to zero? The API is asking if this is
      // "external". Is this asking if it has external linkage? Or is it asking
      // if this is externally defined (as in C's extern)? In either case, why
      // are we always passing 0 here? Is this just the "safer" course, or is it
      // that we just haven't yet encountered a situation where this should be
      // non-zero? Or does cuda require this to be zero currently because it is
      // they who have not implemented something?
      Constant *gExt = ConstantInt::get(i32Ty, 0);

      Value *args[] = {bundleHandle, hostG, gName, gName, gExt, gSize, gConst,
                       // Per the documentation, The last argument to
                       // cudaRegisterVar() must always be zero
                       czero};

      LLVM_DEBUG(dbgs() << "\t\t\tregister global '" << hostG->getName()
                        << "' via ctor runtime call.\n");
      builder.CreateCall(cudaRegisterVar, args);
    }

    // Wrap up fatbinary registration steps.
    FunctionCallee cudaRegisterFatBinaryEnd =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_register_fat_binary_end);
    builder.CreateCall(cudaRegisterFatBinaryEnd, bundleHandle);

    // Now add the dtor to help us clean up at program exit.
    FunctionCallee atExit = getOrInsertLibFunc(&m, tli, LibFunc_atexit);
    builder.CreateCall(atExit, dtor);

    builder.CreateRetVoid();
    return ctor;
  }

  Function *createDtor(Module &m, GlobalVariable *gBundleHandle) {
    const DataLayout &dl = m.getDataLayout();
    Align alignPtr = dl.getPointerABIAlignment(0);

    LLVMContext &ctx = m.getContext();
    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);

    FunctionType *dtorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *dtor = Function::Create(dtorTy, GlobalValue::InternalLinkage,
                                      ".kitcuda.dtor", &m);

    TargetLibraryInfo &tli = getTLI(*dtor);
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", dtor));
    Value *handle = builder.CreateAlignedLoad(ptrTy, gBundleHandle, alignPtr);

    FunctionCallee cudaUnregisterFatBinary =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_unregister_fat_binary);
    builder.CreateCall(cudaUnregisterFatBinary, handle);

    ConstantInt *constTT = createConstInt(TTID::Cuda, ctx);
    FunctionCallee kitrtFinalize =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_finalize);
    builder.CreateCall(kitrtFinalize, {constTT});

    builder.CreateRetVoid();
    return dtor;
  }

public:
  GenerateCtorCuda(detail::GetTLI getTLI, const TTOptions &tto,
                   const detail::GenerateCtorOptions &genCtorOpts)
      : getTLI(getTLI), tto(tto), genCtorOpts(genCtorOpts) {}

  void run(Module &m) {
    GlobalVariable *gFB = getEmbFBGlobal(TTID::Cuda, m);
    assert(gFB && "Could not find global with embedded cuda fat binary");

    GlobalVariable *gBC = getEmbBCGlobal(TTID::Cuda, m);
    assert(gBC && "Could not find global with embedded bitcode");

    std::unique_ptr<Module> devM = parseEmbBCGlobal(*gBC);

    GlobalVariable *gBundle = createBundleGV(m, gFB);
    GlobalVariable *gBundleHandle = createBundleHandleGV(m);
    Function *dtor = createDtor(m, gBundleHandle);
    Function *ctor = createCtor(m, *devM, dtor, gBundle, gBundleHandle);

    // Set the priority of this ctor to be very low so it is one of the last to
    // run.
    appendToGlobalCtors(m, ctor, 65536);
  }
};

} // namespace

void llvm::detail::genCtorCuda(Module &m, detail::GetTLI getTLI,
                               const TTOptions &tto,
                               const detail::GenerateCtorOptions &genCtorOpts) {
  GenerateCtorCuda(getTLI, tto, genCtorOpts).run(m);
}
