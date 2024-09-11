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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"

namespace mlir {
namespace sdy {

void addImportPipeline(OpPassManager& pm, StringRef dumpDirectory) {
  pm.addPass(mlir::sdy::createSaveModuleOpPass(dumpDirectory,
                                               "sdy_module_before_sdy_import"));
  // We need to apply the inliner pass so we have a single main function,
  // otherwise we would need to propagate shardings between call ops and callee
  // functions.
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<func::FuncOp>(createConstantSplitterPass());
  pm.addNestedPass<func::FuncOp>(createAddDataFlowEdgesPass());
  pm.addNestedPass<func::FuncOp>(createApplyShardingConstraintsPass());
  pm.addPass(createShardingGroupUnificationPass());
  pm.addPass(createImportMaximalShardingPass());

  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
  pm.addPass(createCanonicalizerPass(
      /*config=*/config, /*disabledPatterns=*/{},
      /*enabledPatterns=*/{"DedupShardingGroupPattern"}));
  pm.addPass(mlir::sdy::createSaveModuleOpPass(dumpDirectory,
                                               "sdy_module_after_sdy_import"));
}

void registerImportPipeline() {
  PassPipelineRegistration<>(
      "sdy-import-pipeline",
      "Run a sequence of import passes needed as a pre-processing step for "
      "Shardy propagation",
      [](OpPassManager& pm) { return addImportPipeline(pm); });
}

}  // namespace sdy
}  // namespace mlir
