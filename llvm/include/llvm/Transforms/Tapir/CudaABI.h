//===- CudaABI.h - Tapir target for Kitsune's cuda runtime ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
//
//  Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This
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

#ifndef LLVM_TRANSFORMS_TAPIR_CUDA_ABI_H
#define LLVM_TRANSFORMS_TAPIR_CUDA_ABI_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <set>

namespace llvm {

class TTOptions;

/// The tapir target to lower tapir loops to kitsune's cuda runtime. The tapir
/// loops will be converted to GPU kernels.
class CudaABI : public TapirTarget {
public:
  CudaABI(Module &HostM, const TTOptions &TTO, ModuleAnalysisManager &AM);
  ~CudaABI();

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final;
  void preProcessModule() override final;
  bool preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F,
                           bool OutliningTapirLoops) override final;
  void postProcessHelper(Function &F) override final;

  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate, bool IsSpawner,
                              BasicBlock *TFEntry) override final;

  void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry) override final;

  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry) override final;
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry) override final;

  void processSubTaskCall(TaskOutlineInfo &TOI,
                          DominatorTree &DT) override final;

  void postProcessModule() override final;

  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *TL) override final;

private:
  /// Currently, we create a single module into which all tapir loops to be
  /// run on an NVIDIA GPU are outlined. A loop outline processor is created for
  /// each tapir loop which will add the outlined code into this module. This
  /// will eventually be converted to PTX and from there to executable GPU code.
  Module KernelModule;

  ValueToValueMapTy GVMap;

  /// When outlining tapir loops into the \ref KernelModule, we need to generate
  /// a name for the outlined function. This name must be unique. In the absence
  /// of debug information, the computed outlined function name consists of a
  /// fixed base with an integer suffix that is incremented for each tapir loop
  /// that is encountered.
  unsigned NextKernelID = 0;
};

/// The loop outline process for transforming a Tapir parallel loop into a
/// cuda kernel function.
class CudaLoop : public LoopOutlineProcessor {
private:
  /// The name of the kernel into which the loop is outlined.
  std::string KernelName;

  /// For GPU targets, we outline the loop into a separate module. This is that
  /// module.
  Module &KernelModule;

  // Cuda/PTX thread index access.
  Function *CUThreadIdxX = nullptr, *CUThreadIdxY = nullptr,
           *CUThreadIdxZ = nullptr;

  // Cuda/PTX block index and dimensions access.
  Function *CUBlockIdxX = nullptr, *CUBlockIdxY = nullptr,
           *CUBlockIdxZ = nullptr;

  Function *CUBlockDimX = nullptr, *CUBlockDimY = nullptr,
           *CUBlockDimZ = nullptr;

  // Cuda/PTX grid dimensions access.
  Function *CUGridDimX = nullptr, *CUGridDimY = nullptr, *CUGridDimZ = nullptr;

  /// The GlobalValue's used in the loop that is being outlined. This includes
  /// functions, global variables, aliases and ifunc's.
  std::set<GlobalValue *> UsedGlobalValues;

  ValueToValueMapTy &GVMap;

  // Some bookkeeping for reduction variables that would allow recreation of
  // reducer lookup call at the outlined loop call site.
  struct ReducerOutlineLoopCallInfo {
    size_t Size;
    Function *IdFn;
    Function *MergeFn;
  };
  // Maps a subset of OrderedInputs that are reducer variables to
  // ReducerOutlineLoopCallInfo.
  DenseMap<const Value *, ReducerOutlineLoopCallInfo> ReducerInputs;

public:
  /// Create a loop outline processor for the cuda tapir target.
  /// @param M The host module
  /// @param KernelModule The module into which the device code will be outlined
  /// @param KernelName The name of the function in the @ref KernelModule into
  ///                   which the loop is outlined
  /// @param TTOpts The tapir target options
  CudaLoop(Module &M, Module &KernelModule, const std::string &KernelName,
           ValueToValueMapTy &GVMap, const TTOptions &TTOpts);
  ~CudaLoop();

  void fixReducersInKernel(Function *KernelF, Value *ThreadIdx, Value *BlockDim,
                           ValueToValueMapTy &VMap);

  void preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) override;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_CUDA_ABI_H
