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

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"

namespace mlir {
namespace sdy {

namespace {

void addCanonicalizerPass(OpPassManager& pm,
                          ArrayRef<std::string> enabledPatterns) {
  pm.addPass(createCanonicalizerPass(GreedyRewriteConfig(),
                                     /*disabledPatterns=*/{},
                                     /*enabledPatterns=*/enabledPatterns));
}

void runShardyPartitioner(OpPassManager& pm, int& dumpIndex,
                          const ExportOptions& options) {
  InsertExplicitReshardsPassOptions passOptions;
  passOptions.enableFullVersion = options.enableInsertExplicitCollectives;
  passOptions.avoidReshardsOnNamedComputations =
      options.avoidReshardsOnNamedComputations;
  pm.addNestedPass<func::FuncOp>(createInsertExplicitReshardsPass(passOptions));

  if (options.enableInsertExplicitCollectives) {
    pm.addPass(mlir::sdy::createSaveModuleOpPass(
        options.dumpDirectory, "after_explicit_reshards", dumpIndex++));
    addCanonicalizerPass(pm, kReshardLabel);
    pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
    // NOTE: ReshardToCollectives pass above generates all-slice collectives,
    // which during the canonicalizer below may be converted to reduce scatters
    // by potentially fusing with preceeding all-reduces, which are inserted
    // during InsertExplicitReshards pass.
    addCanonicalizerPass(pm, kCollectiveLabel);
    if (options.removeAllGatherReduceScatterForCMV1) {
      pm.addNestedPass<func::FuncOp>(
          createRemoveAllGatherReduceScatterForCMV1Pass());
    }
    pm.addPass(mlir::sdy::createSaveModuleOpPass(
        options.dumpDirectory, "after_partitioner_with_global_shapes",
        dumpIndex++));
  } else {
    addCanonicalizerPass(pm, kCollectiveLabel);
    pm.addPass(mlir::sdy::createSaveModuleOpPass(
        options.dumpDirectory, "after_minimal_partitioner_with_global_shapes",
        dumpIndex++));
  }
}

}  // namespace

void addExportPipeline(OpPassManager& pm, int& dumpIndex,
                       const ExportOptions& options) {
  pm.addNestedPass<func::FuncOp>(createConstantOrScalarMergerPass());
  if (!options.avoidExportForPartitioning) {
    pm.addPass(createRemoveShardingGroupsPass());
    pm.addNestedPass<func::FuncOp>(createShardingConstraintToReshardPass());
  }
  pm.addNestedPass<func::FuncOp>(
      createSinkDataFlowEdgesPass(SinkDataFlowEdgesPassOptions{
          /*sinkDebugShardingOrigins=*/options.dumpShardingOrigins,
          /*sinkDebugPropagationEdgeSharding=*/options.dumpPropagationEdges,
      }));
  if (options.updateNonDivisibleInputOutputShardings) {
    pm.addPass(createUpdateNonDivisibleInputOutputShardingsPass());
  }
  pm.addPass(createCloseShardingsPass());

  // / We dump the module after propagation at this point, since the export
  // passes before are removing internal implementation details of the
  // propagation itself and make the module more readable.
  pm.addPass(mlir::sdy::createSaveModuleOpPass(
      options.dumpDirectory, "after_propagation", dumpIndex++));

  // TODO(enver, tomnatan): Consider having a pipeline specifically for
  // reshards/collectives.
  if (!options.avoidExportForPartitioning) {
    runShardyPartitioner(pm, dumpIndex, options);
  }

  if (options.dumpPropagationEdges || options.dumpShardingOrigins) {
    pm.addPass(createRemovePropagationDebugInfoPass());
  }
  if (!options.keepShardingRules) {
    pm.addNestedPass<func::FuncOp>(createDropShardingRulesPass());
  }
}

void addExportPipeline(OpPassManager& pm, const ExportOptions& options) {
  int dumpIndex = 0;
  addExportPipeline(pm, dumpIndex, options);
}

void registerExportPipeline() {
  PassPipelineRegistration<ExportOptions>(
      "sdy-export-pipeline",
      "Run a sequence of export passes needed as a post-processing step for "
      "Shardy propagation",
      [](OpPassManager& pm, const ExportOptions& options) {
        int dumpIndex = 0;
        return addExportPipeline(pm, dumpIndex, options);
      });
}

}  // namespace sdy
}  // namespace mlir
