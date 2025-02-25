/* Copyright 2025 The Shardy Authors.

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

#include "shardy/round_trip_import/pipelines.h"

#include <cassert>
#include <functional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/round_trip_import/import_backend_func_calls.h"
#include "shardy/round_trip_import/import_callback_custom_calls.h"
#include "shardy/round_trip_import/import_constants.h"
#include "shardy/round_trip_import/import_sdy_custom_calls.h"
#include "shardy/round_trip_import/import_shardy_attrs.h"
#include "shardy/round_trip_import/open_while_free_vars_sharding.h"
#include "shardy/round_trip_import/remove_size_one_axes.h"
#include "shardy/round_trip_import/shard_map_import.h"
#include "stablehlo/transforms/Passes.h"

namespace sdy {
namespace round_trip_import {

namespace {

using ::mlir::func::FuncOp;

void addCommonPreImportPasses(mlir::OpPassManager& pm,
                              bool enableConstantImport) {
  pm.addPass(mlir::createSymbolDCEPass());
  // We import `stablehlo.constant` ops to `sdy.constant` ops so that constants
  // aren't folded in greedy pattern rewriters, which would lift them outside of
  // nested regions (this undoes `WhileLoopConstantSinking` HLO pass).
  // Therefore, this pass needs to be applied after any StableHLO pass that
  // expects `stablehlo.constant`, and before any pass that has a greedy pattern
  // rewriter.
  if (enableConstantImport) {
    pm.addNestedPass<FuncOp>(createImportConstantsPass());
  }
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  config.fold = false;
  config.cseConstants = false;
  pm.addNestedPass<FuncOp>(
      mlir::stablehlo::createStablehloAggressiveSimplificationPass(config));
}

void addCommonPostImportPasses(mlir::OpPassManager& pm) {
  pm.addPass(createImportSdyCustomCallsPass());
  pm.addNestedPass<FuncOp>(createOpenWhileFreeVarsShardingPass());
  pm.addPass(createImportBackendFuncCallsPass());
}

}  // namespace

using ::mlir::PassPipelineRegistration;

void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport) {
  addCommonPreImportPasses(pm, enableConstantImport);
  pm.addPass(createSdyRoundTripImportCallbackCustomCallsPass());
  pm.addPass(createSdyRoundTripImportShardyAttrsPass());
  pm.addPass(createSdyRoundTripShardMapImportPass());
  pm.addPass(createSdyRoundTripRemoveSizeOneAxesPass());
  addCommonPostImportPasses(pm);
}

void registerSdyRoundTripImportPipeline() {
  PassPipelineRegistration<> importPipeline(
      "sdy-round-trip-import-pipeline",
      "Run passes to import a StableHLO module into the SDY (Shardy) dialect.",
      std::bind(addSdyRoundTripImportPipeline, std::placeholders::_1, true));
}

}  // namespace round_trip_import
}  // namespace sdy
