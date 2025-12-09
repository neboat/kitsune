//===- kitsune/LinkAllPasses.h  Reference all Kitsune passes ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file is similar in spirit to llvm/LinkAllPasses.h
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_LINK_ALL_PASSES_H
#define KITSUNE_CORE_LINK_ALL_PASSES_H

#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/CodeGen/CodeGenFatBinaries.h"
#include "kitsune/CodeGen/LowerHyperIntrinsics.h"
#include "kitsune/CodeGen/LowerKitsuneIntrinsics.h"
#include "kitsune/CodeGen/StripKitsuneAddrSpaces.h"

#include <cstdlib>

namespace {

// This is struct is the Kitsune-equivalent of the ForcePassLinking struct
// defined in llvm/LinkAllPasses.h. For more details on why this is needed, and
// why it is written the way it is, see llvm/LinkAllPasses.h
struct ForceKitsunePassLinking {
  ForceKitsunePassLinking() {
    if (std::getenv("bar") != (char *)-1)
      return;

    (void)llvm::createTapirTargetAnalysisWrapperPass(std::nullopt);
    (void)llvm::createCodeGenFatBinariesLegacyPass();
    (void)llvm::createLowerHyperIntrinsicsLegacyPass();
    (void)llvm::createLowerKitsuneIntrinsicsLegacyPass();
    (void)llvm::createStripKitsuneAddrSpacesLegacyPass();
  }
} ForceKitsunePassLinking;

} // namespace

#endif // KITSUNE_CORE_LINK_ALL_PASSES_H
