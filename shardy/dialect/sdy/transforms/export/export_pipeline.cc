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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"

namespace mlir {
namespace sdy {

void addExportPipeline(OpPassManager& pm, StringRef dumpDirectory,
                       bool skipConvertToReshard,
                       bool enableInsertExplicitCollectives,
                       bool keepShardingRules) {
  pm.addNestedPass<func::FuncOp>(createConstantMergerPass());
  pm.addPass(createRemoveShardingGroupsPass());
  if (!skipConvertToReshard) {
    pm.addNestedPass<func::FuncOp>(createShardingConstraintToReshardPass());
  }
  pm.addNestedPass<func::FuncOp>(createSinkDataFlowEdgesPass());
  pm.addNestedPass<func::FuncOp>(
      createUpdateNonDivisibleInputOutputShardingsPass());
  if (enableInsertExplicitCollectives) {
    pm.addPass(createCloseShardingsPass());
  }
  if (!enableInsertExplicitCollectives && !skipConvertToReshard) {
    pm.addNestedPass<func::FuncOp>(
        createTempExplicitReshardsForOptimizationsPass());
  }
  pm.addPass(mlir::sdy::createSaveModuleOpPass(dumpDirectory,
                                               "sdy_module_after_sdy_export"));
  // TODO(enver, tomnatan): Consider having a pipeline specifically for
  // reshards/collectives.
  if (enableInsertExplicitCollectives) {
    pm.addNestedPass<func::FuncOp>(createInsertExplicitReshardsPass());
    // TODO(b/414339524): Canonicalize reshards.
    pm.addPass(mlir::sdy::createSaveModuleOpPass(
        dumpDirectory, "sdy_module_after_insert_explicit_reshards"));
    pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
    // TODO(b/414339524): Canonicalize collectives.
    pm.addPass(mlir::sdy::createSaveModuleOpPass(
        dumpDirectory, "sdy_module_after_reshard_to_collectives"));
  }
  if (!keepShardingRules) {
    pm.addNestedPass<func::FuncOp>(createDropShardingRulesPass());
  }
}

void registerExportPipeline() {
  PassPipelineRegistration<>(
      "sdy-export-pipeline",
      "Run a sequence of export passes needed as a post-processing step for "
      "Shardy propagation",
      [](OpPassManager& pm) { return addExportPipeline(pm); });
}

}  // namespace sdy
}  // namespace mlir
