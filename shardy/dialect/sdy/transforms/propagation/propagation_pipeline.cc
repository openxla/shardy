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

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/auto_partitioner_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"

namespace mlir {
namespace sdy {

namespace {

void populateExportOptions(ExportOptions& options,
                           const PropagationOptions& propOptions) {
  options.keepShardingRules = propOptions.keepShardingRules;
  options.dumpDirectory = propOptions.dumpDirectory.str();
  options.avoidExportForPartitioning = propOptions.avoidExportForPartitioning;
  options.enableInsertExplicitCollectives =
      propOptions.enableInsertExplicitCollectives;
  options.removeAllGatherReduceScatterForCMV1 =
      propOptions.removeAllGatherReduceScatterForCMV1;
  options.dumpShardingOrigins = propOptions.debugShardingOrigins;
  options.dumpPropagationEdges = propOptions.debugPropagationEdgeSharding;
}

}  // namespace

void addPropagationPipeline(OpPassManager& pm, int& dumpIndex,
                            const PropagationOptions& options) {
  addImportPipeline(pm, dumpIndex, options.dumpDirectory, options.skipInline);
  {
    PropagationOptions optionsWithKeepShardingRules = options;
    optionsWithKeepShardingRules.keepShardingRules = true;
    // We intentionally don't increment the dump index here, since this pass
    // might dump 0 to multiple files, and will use a nested dump index.
    pm.addPass(createUserPriorityPropagationPass(optionsWithKeepShardingRules,
                                                 dumpIndex));
  }
  if (options.enableAutoPartitioning) {
    pm.addPass(createSaveModuleOpPass(options.dumpDirectory,
                                      "propagation_before_auto_partitioning",
                                      dumpIndex++));
    AutoPartitionerRegistry::addPasses(pm);
  }
  ExportOptions exportOptions;
  populateExportOptions(exportOptions, options);
  addExportPipeline(pm, dumpIndex, exportOptions);
}

void addPropagationPipeline(OpPassManager& pm,
                            const PropagationOptions& options) {
  int dumpIndex = 1;
  addPropagationPipeline(pm, dumpIndex, options);
}

void registerPropagationPipeline() {
  PassPipelineRegistration<>(
      "sdy-propagation-pipeline",
      "Runs the SDY propagation pass, preceded by a sequence of import passes "
      "needed as a pre-processing step for propagation",
      [](OpPassManager& pm) {
        return addPropagationPipeline(pm);
      });
}

}  // namespace sdy
}  // namespace mlir
