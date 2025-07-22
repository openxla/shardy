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

namespace {

GreedyRewriteConfig getCanonicalizerConfig(bool enableRegionSimplification) {
  return GreedyRewriteConfig()
      .setUseTopDownTraversal(true)
      .setRegionSimplificationLevel(enableRegionSimplification
                                        ? GreedySimplifyRegionLevel::Normal
                                        : GreedySimplifyRegionLevel::Disabled)
      .enableFolding(false)
      .enableConstantCSE(false);
}

}  // namespace

void addImportPipeline(OpPassManager& pm, int& dumpIndex,
                       StringRef dumpDirectory, bool skipInline) {
  // We need to apply the inliner pass so we have a single main function,
  // otherwise we would need to propagate shardings between call ops and callee
  // functions.
  if (!skipInline) {
    pm.addPass(createInlinerPass({}, [&](OpPassManager& pm) {
      pm.addPass(createCanonicalizerPass(
          getCanonicalizerConfig(/*enableRegionSimplification=*/true)));
    }));
  }
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createLiftInlinedMeshesPass());
  pm.addPass(createRemoveSizeOneAxesPass());
  pm.addNestedPass<func::FuncOp>(createConstantOrScalarSplitterPass());
  pm.addPass(createManualAxesCleanupPass());

  // We dump the module before propagation at this point, since the import
  // passes before are cleanup passes that make the module more readable, and
  // the import passes after are internal implementation details that are part
  // of the propagation itself.
  pm.addPass(mlir::sdy::createSaveModuleOpPass(
      dumpDirectory, "before_propagation", dumpIndex++));

  pm.addNestedPass<func::FuncOp>(createAddDataFlowEdgesPass());
  pm.addNestedPass<func::FuncOp>(createApplyShardingConstraintsPass());
  // The sharding group import pass must run after applying sharding
  // constraints. This ensures we can detect sharding conflicts between group
  // members which have pre-propagation shardings due to sharding constraints.
  pm.addPass(createShardingGroupImportPass());
}

void addImportPipeline(OpPassManager& pm, StringRef dumpDirectory,
                       bool skipInline) {
  int dumpIndex = 0;
  addImportPipeline(pm, dumpIndex, dumpDirectory, skipInline);
}

void registerImportPipeline() {
  PassPipelineRegistration<>(
      "sdy-import-pipeline",
      "Run a sequence of import passes needed as a pre-processing step for "
      "Shardy propagation",
      [](OpPassManager& pm) {
        int dumpIndex = 0;
        addImportPipeline(pm, dumpIndex);
      });
}

}  // namespace sdy
}  // namespace mlir
