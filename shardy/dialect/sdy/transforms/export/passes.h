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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_PASSES_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_PASSES_H_

// IWYU pragma: begin_keep

#include <stdbool.h>

#include <memory>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

// IWYU pragma: end_keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

struct ExportOptions : public PassPipelineOptions<ExportOptions> {
  Option<std::string> dumpDirectory{
      *this, "dump-directory",
      llvm::cl::desc("Directory to dump intermediate MLIR modules."),
      llvm::cl::init("")};

  Option<bool> avoidExportForPartitioning{
      *this, "avoid-export-for-partitioning",
      llvm::cl::desc(
          "Avoid exporting the module for partitioning so that the module will "
          "be compatible for another round of propagation."),
      llvm::cl::init(false)};

  Option<bool> enableInsertExplicitCollectives{
      *this, "enable-insert-explicit-collectives",
      llvm::cl::desc("Enable inserting explicit collective ops during export."),
      llvm::cl::init(false)};

  Option<bool> keepShardingRules{
      *this, "keep-sharding-rules",
      llvm::cl::desc("Keep sdy.sharding_rule attrs."), llvm::cl::init(false)};

  Option<bool> dumpShardingOrigins{
      *this, "dump-sharding-origins",
      llvm::cl::desc("Sink sdy.sharding_origins attr."), llvm::cl::init(false)};

  Option<bool> dumpPropagationEdges{
      *this, "dump-propagation-edges",
      llvm::cl::desc("Sink sdy.propagation_edges attr."),
      llvm::cl::init(false)};
};

// Adds a sequence of export passes needed as a post-processing step for SDY
// propagation.
//
// Takes the current index of the module dump, which will be included as a
// prefix in the file name, and incremented after each stage (dumped module).
void addExportPipeline(OpPassManager& pm, int& dumpIndex,
                       const ExportOptions& options = {});

// Same as above, but initializes a default dump index to 0.
void addExportPipeline(OpPassManager& pm, const ExportOptions& options = {});

// Register the sdy-export-pipeline.
void registerExportPipeline();

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_PASSES_H_
