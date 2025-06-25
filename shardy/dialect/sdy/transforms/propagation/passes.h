/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_PASSES_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_PASSES_H_

// IWYU pragma: begin_keep

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

// IWYU pragma: end_keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

struct PropagationOptions {
  // Whether to keep existing and created `sdy::OpShardingRuleAttr` on ops.
  bool keepShardingRules = false;
  // The system directory to dump various rewritten modules for debugging.
  StringRef dumpDirectory = "";
  // Whether to avoid shardings that may cause values to be non-divisible by its
  // dimension sharding.
  bool conservativePropagation = false;
  // Whether to save debug information about the sharding origins on the module.
  bool debugShardingOrigins = false;
  // Whether to save debug information about the edge shardings on the module.
  bool debugPropagationEdgeSharding = false;
  // Whether to avoid exporting the module for partitioning so that the module
  // will be compatible for another round of propagation.
  bool avoidExportForPartitioning = false;
  // Whether to skip inlining in the module.
  bool skipInline = false;
  // Whether to enable inserting explicit collectives.
  bool enableInsertExplicitCollectives = false;
};

// Adds the SDY propagation pass, preceded by a sequence of import passes needed
// as a pre-processing step for propagation.
//
// The added propagation pass is the top-level layer of propagation, which
// includes all conflict resolution strategies in a hierarchy.
void addPropagationPipeline(OpPassManager& pm,
                            const PropagationOptions& options = {});

// Register the sdy-propagation-pipeline.
void registerPropagationPipeline();

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_PASSES_H_
