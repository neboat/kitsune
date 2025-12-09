//===- CudaABI.cpp - Lower Tapir Kitsune's cuda runtime -----------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023, 2025. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Kitsune's cuda runtime
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "kitsune/Core/CommandLineOptions.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/TargetUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TapirLoopHints.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/KitsuneUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "cuabi"

// For some background material see the NVPTX target documentation
// at https://llvm.org/docs/NVPTXUsage.html.
//
// NOTE: We currently do not support the full range of GPU architectures
// supported by the NVPTX backend. This is primarily due to a lack of resources
// to test every GPU.
//
// This transformation outlines a tapir loop into a kernel module that will
// eventually be compiled to NVIDIA GPU code. Calls are added in the original
// module to use Kitsune's cuda runtime to launch the tapir loops. There is a
// lot more that needs to be done before the kernel module can be compiled to
// GPU code, but those steps are handled in subsequent passes.

// This is meant to be a factor used for additional kernel optimizations but is
// currently not used this. It should be left in its default state.
static cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one (default=1)"),
    cl::cat(cl::catKitClDevOpts));

// Enable/Disable flush denorms-to-zero code generation.
static cl::opt<bool> clFTZ("cuabi-ftz", cl::init(false), cl::Hidden,
                           cl::desc("Enable flush-denorms-to-zero"),
                           cl::cat(cl::catKitClDevOpts));

// FIXME: The default is currently set to true. This should be changed to false
// and the name of the option changed.
//
// FIXME: We really should not be exposing command line options from other
// source files. This is an experimental option that has been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
//
// Request that the runtime carry out an extra set of steps to attempt to refine
// the launch parameters of kernels. In this mode of operation the compiler will
// provide some compile-time information to the runtime for assisting in the
// assisting in the analysis and refinement of launches.
cl::opt<bool> clRefineLaunches(
    "cuabi-refine-launches", cl::init(true), cl::Hidden,
    cl::desc("Enable runtime's refinement of launch parameters"),
    cl::cat(cl::catKitClDevOpts));

cl::opt<bool> UseKitCudaRuntimeBC(
    "use-kitcuda-runtime-bc", cl::init(true),
    cl::desc("Use a bitcode file for the Kitsune CUDA runtime ABI"),
    cl::Hidden);

cl::opt<std::string> ClKitCudaRuntimeBCPath(
    "kitcuda-runtime-bc-path", cl::init(""),
    cl::desc("Path to the bitcode file for the Kitsune CUDA runtime ABI"),
    cl::Hidden);

/// This prefix is intentionally *NOT* __kitcuda to ensure that there is no
/// confusion - and, more importantly, no collisions - between any names
/// prefixed with this and the symbols from kitsune's cuda runtime which are
/// typically prefixed with __kitcuda.
static constexpr StringRef CUABI_PREFIX = "__kitcu_";
static constexpr StringRef CUABI_KERNEL_NAME_PREFIX = "__kitcu_loop_";

/// ptxas has several restrictions on the names of symbols, including internal
/// symbols. If the given name is not valid for PTX, return a modified name.
/// Otherwise, just return a clone of the name. The result is prefixed with a
/// string to reduce the likelihood of collisions. This behavior can be
/// overridden by passing false to \ref addPrefix.
static std::string convertNameForPTX(StringRef name, bool addPrefix = true) {
  auto isInvalidChar = [](char c) -> bool { return c == '.' or c == '-'; };
  if (std::none_of(name.begin(), name.end(), isInvalidChar))
    return name.str();

  // Simply replacing the invalid characters with _ may not be safe because
  // there is a chance of collisions with other symbols in the module. In most
  // languages that we care about, a double-underscore at the start of an
  // identifier name is reserved for the compiler, so we prefix the newly
  // created names with such a prefix.
  std::string buf;
  llvm::raw_string_ostream os(buf);
  if (addPrefix)
    os << CUABI_PREFIX << "_nwnm__";
  for (char c : name)
    os << (isInvalidChar(c) ? '_' : c);
  return buf;
}

CudaLoop::CudaLoop(Module &M, Module &KernelModule, const std::string &KN,
                   ValueToValueMapTy &GVMap, const TTOptions &TTOpts)
    : LoopOutlineProcessor(M, KernelModule, TTOpts,
                           CloneFunctionChangeType::DifferentModule),
      KernelName(KN), KernelModule(KernelModule), GVMap(GVMap) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: creating a cuda loop outliner.\n"
                    << "  - target kernel name: " << KernelName << "\n");

  // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);
}

CudaLoop::~CudaLoop() {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: destroying loop outliner for kernel '"
                    << KernelName << "'.\n");
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &TL,
                                   ValueToValueMapTy &FVMap) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: -preprocessing loop for kernel '"
                    << KernelName << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it must be
  // be cloned into the KernelModule and then registered with the cuda runtime.
  // The registration will be done in the global ctor which will be generated by
  // a later pass.
  collectGlobalValues(*TL.getLoop(), UsedGlobalValues);

  // NVPTX has a number of different address spaces. We do not use them and the
  // code seems to work. It is not clear if there is any advantage to using
  // them, but it may be a good idea to look into it at some point.
  cloneUsedGlobalVariablesInto(KernelModule, UsedGlobalValues, GVMap);

  // ptxas imposes restrictions on the names that global entities may have.
  // Ideally, it would be good to do this in a post-processing pass, say the
  // prepare embedded module pass. However, the names of the globals must be
  // passed to Kitsune's intrinsics, so we have to do this here. The alternative
  // would involve an unhealthy amount of value chasing across two different
  // LLVM modules and is almost certainly not worth the trouble.
  for (GlobalValue *v : UsedGlobalValues)
    if (auto *g = dyn_cast<GlobalVariable>(v))
      cast<GlobalVariable>(GVMap[g])->setName(convertNameForPTX(g->getName()));

  // The global variables have to be cloned before cloning the functions because
  // they may be used in the bodies of functions to be cloned.
  cloneReachableFuncsInto(KernelModule, UsedGlobalValues, GVMap, TLI);
  cloneReachableIFuncsInto(KernelModule, UsedGlobalValues, GVMap);

  // The aliasee in global aliases is a global value, so they must be cloned
  // after the global variables and functions are in the vmap.
  cloneUsedGlobalAliasesInto(KernelModule, UsedGlobalValues, GVMap);

  // Copy mappings from global VMap into function-local VMap.
  for (const auto Mapping : GVMap)
    FVMap[Mapping.first] = Mapping.second;
  if (GVMap.hasMD())
    for (const auto &MDMapping : GVMap.MD())
      FVMap.MD()[MDMapping.first] = MDMapping.second;
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                  ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = KernelModule.getContext();
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  TapirLoopHints Hints(TL);

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Type *PrimaryIVType = PrimaryIV->getType();

  // We no longer need the cloned sync region.
  auto *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  Function *KernelF = Out.Outline;

  // Set the generated name for the kernel. This name is passed to the runtime's
  // kernel launch function, so it must be set correctly.
  LLVM_DEBUG(dbgs() << "Renaming KernelF " << KernelF->getName() << " to "
                    << KernelName << "\n");
  KernelF->setName(KernelName);
  LLVM_DEBUG(dbgs() << "  new name: " << KernelF->getName() << "\n");
  // If KernelName was already in use, then KernelF may have a different name
  // than KernelName.  Get the actual name of KernelF.
  KernelName = KernelF->getName();

  // Set the linkage of the kernel to external to prevent it from being DCE'ed
  // since there will be no caller for the function in the kernel module.
  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelF->setCallingConv(CallingConv::PTX_Kernel);

  // Remove all target-related attributes from the kernel function. These may be
  // present because the frontend believes that the code is being compiled for
  // the CPU (host) only.
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("tune-cpu");

  // Remove some functions that are relevant for functionality that is not
  // supported on the GPU. For instance, exceptions are not currently available
  // on GPU's.
  KernelF->removeFnAttr("personality");

  // Add an attribute identifying this as a function outlined from a tapir loop.
  KernelF->addFnAttr(Attribute::KitKernel);

  // Replace some of the target-specific attributes with the correct ones.
  KernelF->addFnAttr("target-cpu", getOptions().getCudaArch());
  KernelF->addFnAttr("target-features",
                     join_items(",", getOptions().getCudaTargetFeatures(),
                                getOptions().getCudaArch()));

  // Add other attributes that are relevant for the target.
  KernelF->addFnAttr("uniform-work-group-size", "true");

  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 6> AV;
  AV.push_back(ValueAsMetadata::get(KernelF));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  // AV.push_back(MDString::get(Ctx, "maxntidx"));
  // AV.push_back(ValueAsMetadata::get(
  //     ConstantInt::get(Type::getInt32Ty(Ctx), MaxThreadsPerBlock)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Verify that the Thread ID corresponds to a valid iteration. Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1. The End argument encodes the loop limit. Get
  // end and grainsize arguments

  // End argument is always the second argument in the kernel function.
  Argument *End = KernelF->getArg(1);

  // Get the grainsize value, which is either constant or the third LC arg.
  // TODO: We only support a grain size of 1 right now. Not clear if this
  // could be a future optimization but strip mining on our current tests only
  // results in degraded performance.
  // if (unsigned ConstGrainsize = TLI.getGrainsize())
  //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
  // else
  Value *Grainsize =
      ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());

  IRBuilder<> B(Entry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic CUDA thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx = B.CreateCall(CUThreadIdxX);
  Value *BlockIdx = B.CreateCall(CUBlockIdxX);
  Value *BlockDim = B.CreateCall(CUBlockDimX);
  Value *BDxBI = B.CreateMul(BlockIdx, BlockDim, "blk_offset");
  Value *TIpBDxBI = B.CreateAdd(ThreadIdx, BDxBI, "cuthread_id");
  Value *ThreadIV =
      B.CreateIntCast(TIpBDxBI, PrimaryIVType, false, "thread_iv");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the current
  // codegen.
  // ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadIV, Grainsize, "thread_end");
  Value *Cond = B.CreateICmpUGE(ThreadIV, End, "cond_thread_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadIV);
  // TODO: ???? PrimaryIVInput->eraseFromParent();

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {
  LLVM_DEBUG(dbgs() << "cudaloop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);

  ConstantInt *CTT = createConstInt(TTID::Cuda, Ctx);
  GlobalVariable *KProps =
      createKernelPropertiesGlobal(KernelName, TTID::Cuda, M);
  Value *KName = createConstString(KernelName, M);
  GlobalVariable *EmbFB = getEmbFBGlobal(TTID::Cuda, M);

  // At this point we need a threads-per-block value for the launch call. The
  // runtime will determine this value if ThreadsPerBlock is zero but it can
  // also be overridden via kitsune's forall launch attribute. The catch here is
  // the launch attribute's value for this is flexible and be a computed
  // expression vs. a compile-time constant. For this first step of creating the
  // kernel launch, we take the path of a runtime configuration vs. an
  // attributed launch.
  TapirLoopHints Hints(TL.getLoop());
  Value *TPB = ConstantInt::get(Int32Ty, Hints.getThreadsPerBlock());

  CallBase *CallOutlined = cast<CallBase>(TOI.ReplCall);
  BasicBlock *RCBB = CallOutlined->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(CallOutlined);
  IRBuilder<> Builder(&NewBB->front());

  // Deal with type mismatches for the trip count.
  Value *TripCount = CallOutlined->getArgOperand(1);
  if (TripCount->getType() != Int64Ty)
    TripCount = Builder.CreateSExtOrBitCast(TripCount, Int64Ty, "cast.tc");

  // We need to explicitly sync non-const globals that are used in the kernel
  // before the kernel is launched.
  copyNonConstGlobalsHToD(UsedGlobalValues, TTID::Cuda, M, Builder);

  Value *CudaStream =
      Builder.CreateIntrinsic(PtrTy, Intrinsic::kit_thread_stream, {CTT});
  std::vector<Value *> Args = {CTT, EmbFB,  KName,     TripCount,
                               TPB, KProps, CudaStream};
  for (Value *Inp : CallOutlined->args())
    Args.push_back(Inp);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  (void)Builder.CreateCall(
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::kit_async_launch_kernel),
      Args);
  (void)Builder.CreateIntrinsic(VoidTy, Intrinsic::kit_sync_stream,
                                {CTT, CudaStream});

  // After the kernel is done, copy the non-const globals back to the host. This
  // is done here to keep this part of the code generation simple. A subsequent
  // pass will attempt to move this call to the point where the globals are
  // actually used on the host (or perhaps even delete it if the host never uses
  // the global again).
  copyNonConstGlobalsDToH(UsedGlobalValues, TTID::Cuda, M, Builder);

  CallOutlined->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

CudaABI::CudaABI(Module &M, const TTOptions &TTO, ModuleAnalysisManager &AM)
    : TapirTarget(M, TTO), KernelModule("", M.getContext()), NextKernelID(0) {
  LLVM_DEBUG(dbgs() << "cuabi: CudaABI::CudaABI()\n");

  TargetMachine *TM = createTargetMachine(TTID::Cuda, TTO);
  KernelModule.setTargetTriple(TM->getTargetTriple());
  KernelModule.setDataLayout(TM->createDataLayout());

  KernelModule.setModuleIdentifier(getNameForDeviceModule(M, CUABI_PREFIX));
  addDeviceModuleMetadata(TTID::Cuda, KernelModule);
  cloneModuleFlagsMetadataInto(M, KernelModule);
  cloneIdentMetadataInto(M, KernelModule);
  KernelModule.setModuleFlag(Module::Override, "nvvm-reflect-ftz", clFTZ);
}

CudaABI::~CudaABI() { LLVM_DEBUG(dbgs() << "cuabi: destroy tapir target.\n"); }

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast than the CPU
  // cases Tapir was originally designed for. At present keeping the grainsize
  // at 1 has almost always shown to yield the best results.  It is obviously
  // not the best choice for all cases...
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize.getValue());
  // Replace uses of grainsize intrinsic call with a computed grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // The CUDA transformation splits the code into two modules, one for the host,
  // the other for the device. The sync instruction will only be present on the
  // host module.
}

void CudaABI::addHelperAttributes(Function &F) {}

void CudaABI::preProcessModule() {
  // Create the global variable that will eventually contain the fat binary of
  // GPU code. This is currently uninitialized, but will be passed to several
  // of the kitsune runtime intrinsic calls when launching kernels, copying
  // global variables from host to device etc.
  (void)createEmbFBGlobal(TTID::Cuda, M);
}

bool CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
  return false;
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {}

void CudaABI::postProcessHelper(Function &F) {}

void CudaABI::preProcessOutlinedTask(Function &, Instruction *, Instruction *,
                                     bool, BasicBlock *) {}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner, BasicBlock *TFEntry) {}

void CudaABI::postProcessRootSpawner(Function &F, BasicBlock *TFEntry) {}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {}

void CudaABI::preProcessRootSpawner(Function &, BasicBlock *TFEntry) {}

void CudaABI::postProcessModule() {
  LLVM_DEBUG(dbgs() << "cuabi: post processing kernel and host modules...\n");

  // Link in the Bitcode file
  if (UseKitCudaRuntimeBC) {
    SMDiagnostic SMD;
    LLVMContext &C = KernelModule.getContext();
    // Parse the bitcode file.  This call imports structure definitions, but not
    // function definitions.
    if (std::unique_ptr<Module> ExternalModule =
            parseIRFile(ClKitCudaRuntimeBCPath, SMD, C)) {
      // Link the external module into the current module, copying over global
      // values.
      bool Fail = Linker::linkModules(KernelModule, std::move(ExternalModule),
                                      Linker::Flags::LinkOnlyNeeded);
      if (Fail)
        C.emitError("CudaABI: Failed to link bitcode ABI file: " +
                    Twine(ClKitCudaRuntimeBCPath));
    } else {
      C.emitError("CudaABI: Failed to parse bitcode ABI file: " +
                  Twine(ClKitCudaRuntimeBCPath));
    }
  }

  if (verifyModule(KernelModule, &errs())) {
    LLVM_DEBUG(dbgs() << "Kernel Module before embedding:" << KernelModule);
    llvm_unreachable("Loop spawning produced bad IR!");
  }

  if (verifyModule(M, &errs())) {
    LLVM_DEBUG(dbgs() << "Host Module before embedding:" << M);
    llvm_unreachable("Loop spawning produced bad IR!");
  }

  // At this point, we are done with the minimum task of outlining the tapir
  // loop into a kernel module. There are still a number of transformations that
  // must be carried out on this module before it can be compiled to GPU code,
  // but those will be done by subsequent passes. The module here is in a state
  // where we can perform combined host/device analyses and optimizations.
  (void)createEmbBCGlobal(KernelModule, TTID::Cuda, M);
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  LLVM_DEBUG(dbgs() << "cuabi: create loop outlining processor.\n");
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".input"));

  std::string KernelName = convertNameForPTX(
      getNameForTapirLoop(*TL, CUABI_KERNEL_NAME_PREFIX, NextKernelID++),
      /*AddPrefix=*/false);
  return new CudaLoop(M, KernelModule, KernelName, GVMap, this->getOptions());
}
