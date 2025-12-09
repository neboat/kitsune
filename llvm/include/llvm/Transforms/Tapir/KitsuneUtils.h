//===- KitsuneUtils.h - Kitsune-specific utilities -------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for the Kitsune-specific tapir targets. Moving these to a Kitsune
// support library is more trouble than it is worth.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_KITSUNE_UTILS_H
#define LLVM_TRANSFORMS_TAPIR_KITSUNE_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <set>

namespace llvm {

class BasicBlock;
class Function;
class GlobalValue;
class Loop;
class Module;
class TapirLoopInfo;

/// Collect the GlobalValues used in a \ref BasicBlock.
/// \param bb The basic block
/// \param seen The set into which to collect the GlobalValue's
void collectGlobalValues(llvm::BasicBlock &bb,
                         std::set<llvm::GlobalValue *> &seen);

/// Collect the GlobalValues used in a \ref Function.
/// \param f The function
/// \param seen The set into which to collect the GlobalValue's
void collectGlobalValues(Function &f, std::set<GlobalValue *> &seen);

/// Get the GlobalValues used in a \ref Loop.
/// \param loop The loop
/// \param seen The set into which to collect the GlobalValue's
void collectGlobalValues(Loop &loop, std::set<GlobalValue *> &seen);

/// Clone global variables that are used in outlined tapir loop from the host to
/// the device module.
/// \param devM The device module
/// \param usedGlobalValues The global values used by the tapir loop
/// \param vmap The ValueMap containing mappings for the global values that are
///             cloned into the device module
/// \param asConstant Address space for constant global variables
/// \param asNonConst Address space for non-constant global variables
/// \param visConst Visibility of constant global variables
/// \param visNonConst Visibility of non-constant global variables
void cloneUsedGlobalVariablesInto(
    Module &devM, const std::set<GlobalValue *> &usedGlobalValues,
    ValueToValueMapTy &vmap, unsigned asConst = 0, unsigned asNonConst = 0,
    GlobalValue::VisibilityTypes visConst = GlobalValue::DefaultVisibility,
    GlobalValue::VisibilityTypes visNonConst = GlobalValue::DefaultVisibility);

/// Clone functions that are transitively reachable from outlined tapir loops
/// from the host to the device module.
/// \param devM The device module
/// \param usedGlobalValues The global values used by the tapir loop
/// \param vmap The ValueMap containing mappings for the global values that are
///             cloned into the device module
void cloneReachableFuncsInto(Module &devM,
                             const std::set<GlobalValue *> &usedGlobalValues,
                             ValueToValueMapTy &vmap,
                             const TargetLibraryInfo &TLI);

/// Clone IFuncs that are transitively reachable from outlined tapir loops from
/// the host to the device module.
/// \param devM The device module
/// \param usedGlobalValues The global values used by the tapir loop
/// \param vmap The ValueMap containing mappings for the global values that are
///             cloned into the device module
void cloneReachableIFuncsInto(Module &devM,
                              const std::set<GlobalValue *> &usedGlobalValues,
                              ValueToValueMapTy &vmap);

/// Clone global aliases that are used in outlined tapir loop from the host to
/// the device module.
/// \param devM The device module
/// \param usedGlobalValues The global values used by the tapir loop
/// \param vmap The ValueMap containing mappings for the global values that are
///             cloned into the device module
void cloneUsedGlobalAliasesInto(Module &devM,
                                const std::set<GlobalValue *> &usedGlobalValues,
                                ValueToValueMapTy &vmap);

/// Construct the name to be used for the outlined function consisting of the
/// body of the tapir loop.
std::string getNameForTapirLoop(const TapirLoopInfo &tl, StringRef prefix = "",
                                unsigned suffix = 0);

/// Construct the name for a device module.
std::string getNameForDeviceModule(const Module &hostM, StringRef prefix = "");

/// Generate calls to copy non-constant globals, which are used in the outlined
/// tapir loop, from device to host after launching the kernel.
void copyNonConstGlobalsDToH(const std::set<GlobalValue *> &gvs, TTID tt,
                             Module &m, IRBuilder<> &builder);

/// Generate calls to copy non-constant globals, which are used in the outlined
/// tapir loop, from host to device before launching the kernel.
void copyNonConstGlobalsHToD(const std::set<GlobalValue *> &gvs, TTID tt,
                             Module &m, IRBuilder<> &builder);

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_KITSUNE_UTILS_H
