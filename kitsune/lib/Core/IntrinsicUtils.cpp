//===- IntrinsicUtils.cpp - Utilities for Kitsune's intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune's intrinsics
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IntrinsicUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

bool llvm::isKitsuneIntrinsic(Intrinsic::ID id) {
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.");
}

bool llvm::isKitsuneIntrinsicAsync(Intrinsic::ID id) {
  assert(isKitsuneIntrinsic(id) && "Must be a kitsune intrinsic");
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.async.");
}

bool llvm::isKitsuneIntrinsicBlocking(Intrinsic::ID id) {
  return not isKitsuneIntrinsicAsync(id);
}

bool llvm::isHyperIntrinsic(Intrinsic::ID id) {
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.reducer");
}

Value *llvm::getStreamFromLaunch(const CallBase &call) {
  assert(call.getIntrinsicID() == Intrinsic::kit_async_launch_kernel &&
         "Instruction must call async_launch_kernel intrinsic");

  // The last parameter of the function type of the callee is the "var arg
  // type". By definition, this is also the argument number of the first
  // variadic argument in the call. The argument immediately before this is the
  // stream.
  Function *callee = call.getCalledFunction();
  FunctionType *calleeTy = callee->getFunctionType();
  return call.getArgOperand(calleeTy->getNumParams() - 1);
}

std::vector<Value *> llvm::getKernelArgumentsFromLaunch(const CallBase &call) {
  assert(call.getIntrinsicID() == Intrinsic::kit_async_launch_kernel &&
         "Instruction must call async_launch_kernel intrinsic");

  // The last parameter of the function type of the callee is the "var arg
  // type". By definition, this is also the argument number of the first
  // variadic argument in the call. This, along with all subsequent arguments
  // in the call are the arguments to the kernel function being launched.
  std::vector<Value *> args;
  Function *callee = call.getCalledFunction();
  FunctionType *calleeTy = callee->getFunctionType();
  for (unsigned i = calleeTy->getNumParams(); i < call.arg_size(); ++i)
    args.push_back(call.getArgOperand(i));

  return args;
}
