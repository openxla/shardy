/* Copyright 2025 The MPMD Authors.

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
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mlir::mpmd {

void addShardingPropagationPipeline(OpPassManager& pm,
                                    llvm::StringRef sdyDumpDir) {
  // Uniquify function inputs and outputs, in case the same fragment result or
  // function input is returned multiple times with different shardings.
  UniquifyFunctionInputsOutputsPassOptions uniquifyOptions;
  uniquifyOptions.useTransferInsteadOfFragment = true;
  pm.addNestedPass<func::FuncOp>(
      createUniquifyFunctionInputsOutputsPass(uniquifyOptions));

  // Run sdy propagation.
  sdy::PropagationOptions options;
  options.dumpDirectory = sdyDumpDir;
  options.avoidExportForPartitioning = true;
  options.skipInline = true;
  sdy::addPropagationPipeline(pm, options);

  // Enforce user specified shardings.
  pm.addNestedPass<func::FuncOp>(createEnforceUserShardingsPass());

  // Extract reshard from inter-mesh transfers.
  pm.addNestedPass<func::FuncOp>(
      createExtractReshardsFromInterMeshTransfersPass());

  // Add the shardings back to `MeshTensorType`. Before this pass, the shardings
  // are on the attributes of fragments and transfer ops.
  pm.addNestedPass<func::FuncOp>(createConvertSdyShardingsToMpmdTypesPass());

  // No need to keep SDY constants, which prevent constant folding, since we are
  // stripping shardings away from constants in
  // `mpmd-convert-sdy-shardings-to-mpmd-types`.
  pm.addNestedPass<func::FuncOp>(createConvertSdyConstantsPass());

  // Try folding/optimizing to minimize the number of tensors passed across
  // fragments. This is applied here so that later passes like
  // mpmd-fragment-dce can remove unnecessary ops.
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloTargetIndependentOptimizationPass());
}

void registerShardingPropagationPipeline() {
  PassPipelineRegistration<>(
      "mpmd-sharding-propagation-pipeline",
      "Run the standard set of passes to propagate shardings in an "
      "MPMD program.",
      [](OpPassManager& pm) {
        return addShardingPropagationPipeline(pm, /*sdyDumpDir=*/"");
      });
}

}  // namespace mlir::mpmd
