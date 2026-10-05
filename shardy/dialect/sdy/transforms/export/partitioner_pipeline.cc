/* Copyright 2026 The Shardy Authors.

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

#include "shardy/dialect/sdy/transforms/export/partitioner_pipeline.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/transforms/common/partitioner_stage.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"

namespace mlir {
namespace sdy {

void addPartitionerPipeline(OpPassManager& pm,
                            const PartitionerPipelineOptions& options) {
  if (options.stage >= PartitionerStage::kResolvePermutationFactors) {
    ShardyResolvePermutationFactorsPassOptions resolveFactorsOptions;
    resolveFactorsOptions.enableHaloExchange = options.enableHaloExchange;
    resolveFactorsOptions.replicaCount = options.replicaCount;
    resolveFactorsOptions.partitionCount = options.partitionCount;
    resolveFactorsOptions.rngBitGeneratorUnsafe = options.rngBitGeneratorUnsafe;
    pm.addPass(
        createShardyResolvePermutationFactorsPass(resolveFactorsOptions));
  }

  if (options.stage >= PartitionerStage::kReshardToCollectives) {
    pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
  }

  if (options.stage >= PartitionerStage::kOptimizeCollectives) {
    pm.addNestedPass<func::FuncOp>(createOptimizeCollectivesPass());
    // NOTE: run the canonicalizer to fuse all-slice collectives and preceding
    // all-reduces into reduce-scatters, fold back-to-back all-to-alls, and
    // remove no-op collectives, which ConvertGlobalToLocal requires.
    pm.addPass(createCanonicalizerPass(GreedyRewriteConfig(),
                                       /*disabledPatterns=*/{},
                                       /*enabledPatterns=*/{kCollectiveLabel}));
  }

  if (options.stage >= PartitionerStage::kPadForDivisibility) {
    pm.addPass(createPadForDivisibilityPass());
  }

  if (options.stage >= PartitionerStage::kResolveSingleDeviceSharding) {
    ResolveSingleDeviceShardingPassOptions singleDeviceOptions;
    singleDeviceOptions.replicaCount = options.replicaCount;
    singleDeviceOptions.partitionCount = options.partitionCount;
    pm.addNestedPass<func::FuncOp>(
        createResolveSingleDeviceShardingPass(singleDeviceOptions));
  }

  if (options.stage >= PartitionerStage::kConvertGlobalToLocal) {
    ConvertGlobalToLocalPassOptions convertOptions;
    convertOptions.replicaCount = options.replicaCount;
    convertOptions.partitionCount = options.partitionCount;
    pm.addPass(createConvertGlobalToLocalPass(convertOptions));
    DropShardingAndMeshPassOptions dropOptions;
    dropOptions.keepModuleShardingMetadata = true;
    pm.addPass(createDropShardingAndMeshPass(dropOptions));
  }
}

}  // namespace sdy
}  // namespace mlir
