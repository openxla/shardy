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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_PASSES_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_PASSES_H_

// IWYU pragma: begin_keep

#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/transforms/common/distributed_function_pass.h"
#include "shardy/dialect/mpmd/transforms/import/infer_mesh_assignment.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"
#include "shardy/dialect/mpmd/transforms/import/sharding_constraints.h"

// IWYU pragma: end_keep

namespace mlir::mpmd {

// Returns the maximum number of errors to emit when applying a single
// validation pass in mesh inference given `error_limit`. If set to -1, emit all
// errors. Cannot be 0. When set to -1, this also enables other verbose logging.
int GetValidatedMaxErrors(int error_limit);

// Options for the import pipeline.
struct ImportOptions {
  // Mapping between names (of computations and tensors) and mesh names, and
  // optionally stage ids
  UserAssignmentMapOption nameToMeshAssignment;
  // Mapping between function input indices and assigned mesh names.
  IndexedAssignmentMapOption inputIndexToMeshAssignment;
  // Mapping between function output indices and assigned mesh names.
  IndexedAssignmentMapOption outputIndexToMeshAssignment;
  // Constraints enforcing inputs and outputs to be assigned to the same mesh.
  SmallVector<InputOutputEquishardingConstraint> inputOutputConstraints;
  // Whether to merge inferred fragments only after scheduling.
  bool mergeAfterScheduling = false;
  // Whether to absorb inferred fragments into user-defined fragments on
  // entry-point functions.
  bool absorbInferredFragmentsOnEntryPointFunction = false;
  // Whether to clone inferred fragments when merging.
  bool cloneInferredFragments = false;
  // Infer mesh pipeline options.
  InferMeshOptions inferMeshOptions;
  // Enable heterogeneous meshes.
  bool enableHeterogeneousMeshes = false;
  // Whether to split backward fragments.
  bool splitBwdFragments = false;
  // Whether to verify if merging created the right number of scheduling units.
  bool verifyScheduleUnits = false;
};

// Adds the standard set of passes to import an MPMD program with a fixed mesh
// assignment map.
void addImportPipeline(OpPassManager& pm, ImportOptions options);

// Register the `-mpmd-import-pipeline`.
void registerImportPipeline();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_PASSES_H_
