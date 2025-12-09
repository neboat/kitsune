//===- KitsuneUtils.cpp - Kitsune-specific utilities ----------------------===//
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

#include "llvm/Transforms/Tapir/KitsuneUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

using namespace llvm;

static void collectGlobalValues(Constant &c, std::set<GlobalValue *> &seen);

static void collectGlobalValues(GlobalVariable &g,
                                std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  if (g.hasInitializer())
    collectGlobalValues(*g.getInitializer(), seen);
}

static void collectGlobalValues(GlobalIFunc &g, std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
}

static void collectGlobalValues(GlobalAlias &g, std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GlobalAlias not yet supported");
}

static void collectGlobalValues(BlockAddress &blkaddr,
                                std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collectGlobalValues(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collectGlobalValues(*bb, seen);
}

static void collectGlobalValues(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return;

  if (auto *f = dyn_cast<Function>(&c))
    return collectGlobalValues(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collectGlobalValues(*blkaddr, seen);
  else
    for (Use &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collectGlobalValues(*cop, seen);
}

void llvm::collectGlobalValues(BasicBlock &bb, std::set<GlobalValue *> &seen) {
  for (Instruction &inst : bb)
    for (Use &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        ::collectGlobalValues(*c, seen);
}

void llvm::collectGlobalValues(Function &f, std::set<GlobalValue *> &seen) {
  seen.insert(&f);
  for (BasicBlock &bb : f)
    collectGlobalValues(bb, seen);
}

void llvm::collectGlobalValues(Loop &loop, std::set<GlobalValue *> &seen) {
  // Collect the globals used in any subloops.
  for (Loop *subLoop : loop)
    for (BasicBlock *bb : subLoop->blocks())
      collectGlobalValues(*bb, seen);

  // Collect the globals used within the loop itself.
  for (BasicBlock *bb : loop.blocks())
    collectGlobalValues(*bb, seen);
}

void llvm::cloneUsedGlobalVariablesInto(
    Module &devM, const std::set<GlobalValue *> &usedGlobalValues,
    ValueToValueMapTy &vmap, unsigned asConst, unsigned asNonConst,
    GlobalValue::VisibilityTypes visConst,
    GlobalValue::VisibilityTypes visNonConst) {
  for (GlobalValue *v : usedGlobalValues) {
    auto *g = dyn_cast<GlobalVariable>(v);
    if (not g)
      continue;

    assert(g->getType()->getAddressSpace() == 0 &&
           "Global variables must be in default address space");

    // It would be nice to do this as a "post-processing" pass, for instance,
    // when preparing the kernel module for PTX. However, we need to pass the
    // name of the global as a string to several Kitsune intrinsics. That would
    // make the job of renaming them later much more complicated since we would
    // have to modify all the calls in the host that use the global variable's
    // name.
    StringRef name = g->getName();
    bool isConst = g->isConstant();
    Type *type = g->getValueType();
    MaybeAlign align = g->getAlign();
    GlobalValue::ThreadLocalMode threadLocalMode = g->getThreadLocalMode();

    GlobalVariable *newg = nullptr;
    if ((newg = devM.getGlobalVariable(name, /*AllowLocal=*/true))) {
      // If a global with the name is already present in the kernel module,
      // another outlined loop in the host module used the same global. The
      // global is already present, so we just need to update VMap correctly.
      // This is done after this if-else block.
    } else if (isConst) {
      // If the global variable is a constant we can clone it into the device
      // module along with its initializer where it will be treated as an
      // internal variable. There is no coordination with the host.
      Constant *init = g->getInitializer();
      newg = new GlobalVariable(
          devM, type, isConst, GlobalValue::InternalLinkage, init, name,
          /*InsertBefore=*/nullptr, threadLocalMode, asConst);
      newg->setDSOLocal(true);
      newg->setAlignment(align);
      newg->setVisibility(visConst);
    } else {
      // If the global is not constant, we will need to create a device-side
      // version that will have the host-side value copied over prior to
      // launching the kernel.
      Constant *init = Constant::getNullValue(type);
      newg = new GlobalVariable(
          devM, type, isConst, GlobalValue::ExternalLinkage, init, name,
          /*InsertBefore=*/nullptr, threadLocalMode, asNonConst);
      newg->setDSOLocal(true);
      newg->setAlignment(align);
      newg->setVisibility(visNonConst);
    }

    // This is really just a sanity check in case the code above changes and
    // someone makes a silly mistake.
    assert(newg && "All global variables must have a corresponding global "
                   "in the kernel module");

    // The global variables are assumed to be in the default address space when
    // outlining. All uses of the global expect them to be in the default
    // address space. If they are not, cast them in the vmap so when we clone
    // any entities that use them, we do not have type mismatches.
    if (newg->getType()->getAddressSpace()) {
      LLVMContext &ctx = devM.getContext();
      PointerType *ptrTy = PointerType::getUnqual(ctx);
      vmap[g] = ConstantExpr::getAddrSpaceCast(newg, ptrTy);
    } else {
      vmap[g] = newg;
    }
  }
}

void llvm::cloneReachableFuncsInto(
    Module &devM, const std::set<GlobalValue *> &usedGlobalValues,
    ValueToValueMapTy &vmap) {
  // Functions that are called from the tapir loop must be cloned into the
  // kernel module, especially if they contain a body. This is a two-step
  // process - first we create a declaration for the functions since these may
  // be called by the other reachable functions. The VMap already contains
  // mappings for the global variables that may be needed.
  for (GlobalValue *g : usedGlobalValues) {
    if (auto *f = dyn_cast<Function>(g)) {
      std::string fname(f->getName());
      Function *devf = cast<Function>(
          devM.getOrInsertFunction(fname, f->getFunctionType()).getCallee());
      for (unsigned i = 0; i < f->arg_size(); ++i) {
        Argument *a = f->getArg(i);
        Argument *deva = devf->getArg(i);
        deva->setName(a->getName());
        vmap[a] = deva;
      }
      vmap[f] = devf;
    }
  }

  // The vmap now contains mappings from all functions in the source module to
  // their counterparts in the device module. It is now safe to clone the bodies
  // of the functions.
  for (GlobalValue *g : usedGlobalValues) {
    if (auto *f = dyn_cast<Function>(g)) {
      if (f->size() and not f->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> returns;
        auto *devf = cast<Function>(vmap[f]);
        if (devf->size() == 0) {
          CloneFunctionInto(devf, f, vmap,
                            CloneFunctionChangeType::DifferentModule, returns);
          devf->addFnAttr(Attribute::KitDevice);
        }
      }
    }
  }
}

void llvm::cloneReachableIFuncsInto(
    Module &devM, const std::set<GlobalValue *> &usedGlobalValues,
    ValueToValueMapTy &vmap) {
  // IFunc's are a GNU extension, and it is unlikely that we will ever compile
  // code that uses them.
  for (GlobalValue *v : usedGlobalValues)
    if (isa<GlobalIFunc>(v))
      llvm_unreachable("cloneReachableIFuncsInto: not yet implemented");
}

void llvm::cloneUsedGlobalAliasesInto(
    Module &devM, const std::set<GlobalValue *> &usedGlobalValues,
    ValueToValueMapTy &vmap) {
  // FIXME: At some point, we should support global aliases, but right now,
  // there are a number of other features that need to be supported.
  for (GlobalValue *v : usedGlobalValues)
    if (isa<GlobalAlias>(v))
      llvm_unreachable("cloneUsedGlobalAliasesInto: not yet implemented");
}

std::string llvm::getNameForTapirLoop(const TapirLoopInfo &tl, StringRef pfx,
                                      unsigned suffix) {
  std::string buf;
  raw_string_ostream os(buf);
  const Loop *loop = tl.getLoop();
  const Function *f = loop->getHeader()->getParent();
  const Module *m = f->getParent();

  os << pfx;
  if (m->getNamedMetadata("llvm.dbg.cu") || m->getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use the line number to name the
    // kernel. This is only to make debugging a shade easier since it makes it
    // easier to associate the kernel function with a loop in source code.
    //
    // FIXME: This is risky. In principle, in a large project, we could have
    // multiple files with the same name in different directories. There is a
    // small possibility that a forall loop occurs on exactly the same line in
    // both of these files. Ideally, we should include the full file path which
    // is guaranteed to be unique. However, that would detract from the
    // "usefulness" of this name (mainly for debugging). For now, we'll stick
    // with this until we can make some of the support tooling more robust to
    // allow us to mangle the name to avoid collisions.
    DebugLoc loc = loop->getStartLoc();
    unsigned line = loc.getLine();
    unsigned col = loc.getCol();
    StringRef filePath = loc->getFile()->getFilename();
    StringRef fileName = sys::path::filename(filePath);
    os << fileName << "_" << line << "_" << col;
  } else {
    StringRef name = f->getName();
    std::string demangledName;
    if (nonMicrosoftDemangle(name, demangledName,
                             /*CanHaveLeadingDot=*/false,
                             /*ParseParams=*/false))
      os << demangledName;
    else
      os << name;
    os << "_" << suffix;
  }

  return buf;
}

std::string llvm::getNameForDeviceModule(const Module &hostM, StringRef pfx) {
  return join_items("", pfx, sys::path::filename(hostM.getName()));
}

static void copyNonConstGlobals(const std::set<GlobalValue *> &gvs, TTID tt,
                                Intrinsic::ID copyFn, Module &m,
                                IRBuilder<> &builder) {
  const DataLayout &dl = m.getDataLayout();
  LLVMContext &ctx = m.getContext();
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *voidTy = Type::getVoidTy(ctx);
  PointerType *ptrTy = PointerType::getUnqual(ctx);

  GlobalVariable *fb = getEmbFBGlobal(tt, m);
  assert(fb && "Embedded fat binary must exist");

  Constant *ctt = createConstInt(tt, ctx);
  for (GlobalValue *gv : gvs) {
    if (auto *g = dyn_cast<GlobalVariable>(gv)) {
      if (not g->isConstant()) {
        GlobalVariable *name = createConstString(g->getName(), m);
        Type *type = g->getValueType();
        size_t size = dl.getTypeAllocSize(type);
        Constant *bytes = ConstantInt::get(i64Ty, size);

        Value *devPtr = builder.CreateIntrinsic(
            ptrTy, Intrinsic::kit_symbol_device_ptr, {ctt, fb, name});
        if (copyFn == Intrinsic::kit_symbol_memcpy_dtoh)
          (void)builder.CreateIntrinsic(voidTy, copyFn,
                                        {ctt, g, devPtr, bytes});
        else if (copyFn == Intrinsic::kit_symbol_memcpy_htod)
          (void)builder.CreateIntrinsic(voidTy, copyFn,
                                        {ctt, devPtr, g, bytes});
        else
          llvm_unreachable("copyNonConstGlobals: Invalid intrinsic");
      }
    }
  }
}

void llvm::copyNonConstGlobalsDToH(const std::set<GlobalValue *> &gvs, TTID tt,
                                   Module &m, IRBuilder<> &builder) {
  copyNonConstGlobals(gvs, tt, Intrinsic::kit_symbol_memcpy_dtoh, m, builder);
}

void llvm::copyNonConstGlobalsHToD(const std::set<GlobalValue *> &gvs, TTID tt,
                                   Module &m, IRBuilder<> &builder) {
  copyNonConstGlobals(gvs, tt, Intrinsic::kit_symbol_memcpy_htod, m, builder);
}
