//===- LowerHyperIntrinsics.h - Lower hyperobject intrinsics ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers kitsune's hyperobject intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CODEGEN_LOWER_HYPER_INTRINSICS_H
#define KITSUNE_CODEGEN_LOWER_HYPER_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

/// This pass is responsible for lowering Kitsune's hyperobject intrinsics.
class LowerHyperIntrinsicsPass
    : public PassInfoMixin<LowerHyperIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

ModulePass *createLowerHyperIntrinsicsLegacyPass();

} // end namespace llvm

#endif // KITSUNE_CODEGEN_LOWER_HYPER_INTRINSICS_H
