//=- TapirLoopHints.h - Utilities for metadata on tapir loops ----*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for hints on tapir loops
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TAPIR_LOOP_HINTS_H
#define LLVM_ANALYSIS_TAPIR_LOOP_HINTS_H

#include "kitsune/Config/config.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"

#include <map>
#include <variant>

namespace llvm {

class Loop;
class MDNode;
class Metadata;

/// Utility class for getting and setting Tapir-related loop hints in the form
/// of loop metadata.
///
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
class TapirLoopHints {
private:
  /// Alternative with the possible types that a hint can assume.
  using ValueType =
      std::variant<bool, unsigned, TapirSpawnStrategy, std::optional<TTID>>;
  using ValueToValueMapTy = ValueMap<const Value *, WeakTrackingVH>;
  using Hints = std::map<StringRef, ValueType>;

  /// The required prefix on all tapir loop metadata.
  static constexpr StringRef namePrefix = "tapir.loop.";

  // The names of the various hints. These are exactly they appear in the IR.
  // They *MUST* start with the prefix, tapir.loop.
  static constexpr StringRef nameStrategy = "tapir.loop.spawn.strategy";
  static constexpr StringRef nameGrainSize = "tapir.loop.grainsize";
  static constexpr StringRef nameLoopTarget = "tapir.loop.target";
  static constexpr StringRef nameThreadsPerBlock =
      "tapir.loop.threads.per.block";
  static constexpr StringRef nameAutotuneLaunch = "tapir.loop.autotune.launch";
  static constexpr StringRef nameDeferredSync = "tapir.loop.deferred.sync";

  /// All tapir loop hints. Every known loop hint must contain an entry in this
  /// map, even if a hint is not found in the metadata. When adding support for
  /// a new hint, a default value *MUST* be added to this map.
  Hints hints = {{nameStrategy, defaultTapirSpawnStrategy},
                 {nameGrainSize, defaultTapirGrainSize},
                 {nameLoopTarget, defaultTapirTarget},
                 {nameThreadsPerBlock, 0U},
                 {nameAutotuneLaunch, false},
                 {nameDeferredSync, false}};

  /// Check if the value can be serialized to metadata. Some hints cannot
  /// currently be serialized - for instance, those with optional values when
  /// the value is std::nullopt. The name is passed in case we the ability to
  /// serialize depends on the specific hint.
  bool canCreateMetadata(StringRef name, const ValueType &v) const;

  /// Convert the value to one that can be serialized to LLVM's metadata.
  /// Currently, all hints are serialized as unsigned integers, but we may
  /// want to serialize differently depending on the hint. The name is passed
  /// in case certain hints are to be serialized differently even if their
  /// value types are identical.
  ///
  /// This should only be called if the hint can be serialized to LLVM
  /// metadata.
  unsigned toMetadataValue(StringRef name, const ValueType &v) const;

  /// Validate the given value associated with a name in LLVM metadata.
  /// Currently, all hints are serialized as unsigned integers in LLVM metadata,
  /// but this should be changed.
  static bool validate(StringRef name, unsigned v);

public:
  TapirLoopHints(const Loop *loop) : theLoop(loop) {
    // Populate values with existing loop metadata.
    getHintsFromMetadata();
  }

  TapirSpawnStrategy getStrategy() const {
    return std::get<TapirSpawnStrategy>(hints.at(nameStrategy));
  }

  unsigned getGrainsize() const {
    return std::get<unsigned>(hints.at(nameGrainSize));
  }

  std::optional<TTID> getLoopTarget() const {
    return std::get<std::optional<TTID>>(hints.at(nameLoopTarget));
  }

  unsigned getThreadsPerBlock() const {
    return std::get<unsigned>(hints.at(nameThreadsPerBlock));
  }

  bool getAutotuneLaunch() const {
    return std::get<bool>(hints.at(nameAutotuneLaunch));
  }

  bool getDeferredSync() const {
    return std::get<bool>(hints.at(nameDeferredSync));
  }

  /// Clear Tapir hints from the loop's metadata.
  void clearHintsMetadata();

  /// Mark the loop as having no spawning strategy.
  void clearStrategy() {
    hints[nameStrategy] = defaultTapirSpawnStrategy;
    writeHintsToMetadata({{nameStrategy, defaultTapirSpawnStrategy}});
  }

  void clearClonedLoopMetadata(ValueToValueMapTy &VMap) {
    writeHintsToClonedMetadata({{nameStrategy, defaultTapirSpawnStrategy}},
                               VMap);
  }

  void setAlreadyStripMined() {
    hints[nameGrainSize] = 1U;
    writeHintsToMetadata({{nameGrainSize, 1U}});
  }

private:
  /// Find hints specified in the loop metadata and update local values.
  void getHintsFromMetadata();

  /// Set the value of the hint with the given name.
  void setHint(StringRef name, ValueType val);

  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef name, Metadata *arg);

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef name, unsigned v) const;

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *node, const Hints &hints) const;

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(const Hints &hints);

  /// Sets hints into cloned loop metadata, keeping other values intact.
  void writeHintsToClonedMetadata(const Hints &hints, ValueToValueMapTy &vmap);

  /// The loop these hints belong to.
  const Loop *theLoop;
};

/// Returns true if the hints on a tapir loop require outlining during
/// lowering.
bool hintsDemandOutlining(const TapirLoopHints &hints);

} // namespace llvm

#endif // LLVM_ANALYSIS_TAPIR_LOOP_HINTS_H
