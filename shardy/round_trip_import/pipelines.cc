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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/round_trip_import/import_backend_func_calls.h"
#include "shardy/round_trip_import/import_callback_custom_calls.h"
#include "shardy/round_trip_import/import_sdy_custom_calls.h"
#include "shardy/round_trip_import/import_shardy_attrs.h"
#include "shardy/round_trip_import/shard_map_import.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace sdy {

void addSdyRoundTripImportPipeline(OpPassManager& pm) {
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
  config.fold = false;
  config.cseConstants = false;
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloAggressiveSimplificationPass(config));
  pm.addPass(createSdyRoundTripImportCallbackCustomCallsPass());
  pm.addPass(createSdyRoundTripImportShardyAttrsPass());
  pm.addPass(createSdyRoundTripShardMapImportPass());
  pm.addPass(createImportSdyCustomCallsPass());
  pm.addPass(createImportBackendFuncCallsPass());
}

void registerSdyRoundTripImportPipeline() {
  PassPipelineRegistration<> importPipeline(
      "sdy-round-trip-import-pipeline",
      "Run passes to import a StableHLO module into the SDY (Shardy) dialect.",
      addSdyRoundTripImportPipeline);
}

void registerAllSdyRoundTripImportPassesAndPipeline() {
  registerImportSdyCustomCallsPass();
  registerImportBackendFuncCallsPass();
  registerSdyRoundTripImportCallbackCustomCallsPass();
  registerSdyRoundTripImportShardyAttrsPass();
  registerSdyRoundTripShardMapImportPass();

  registerSdyRoundTripImportPipeline();
}

}  // namespace sdy
}  // namespace mlir
