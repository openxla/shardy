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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_PASSES_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_PASSES_H_

// IWYU pragma: begin_keep

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/transforms/common/distributed_function_pass.h"

// IWYU pragma: end_keep

namespace mlir::mpmd {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

// Options for the export pipeline.
struct ExportOptions {
  // Whether to copy constants produced in one fragment to their consumers,
  // possibly through transfers.
  bool copyConstantsFromProducerToConsumer = false;
  // Whether to do more aggressive fragment grouping, across meshes. This may
  // not be desirable for heterogeneous systems.
  bool groupFragmentsAcrossMeshes = true;
  // Whether to apply the merge transfers optimization pass.
  bool applyMergeTransfers = false;
  bool failOnReshardOnlyFragments = false;
  // Whether to enable verbose logging.
  bool verboseLogging = false;
};

// Adds the standard set of passes to export an MPMD program.
void addExportPipeline(OpPassManager& pm, const ExportOptions& options = {});

// Register the `-mpmd-export-pipeline`.
void registerExportPipeline();

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_PASSES_H_
