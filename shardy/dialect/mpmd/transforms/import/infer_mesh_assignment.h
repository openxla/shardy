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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_INFER_MESH_ASSIGNMENT_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_INFER_MESH_ASSIGNMENT_H_

#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/transforms/import/sharding_constraints.h"

namespace mlir::mpmd {

// Options for the infer mesh pipeline.
struct InferMeshOptions {
  // Whether to create transfers when needed, instead of erroring.
  bool inferTransfers = false;
  // Whether to infer cross-mesh reductions.
  bool inferCrossMeshReductions = false;
  // How many copies of a meshless operation we allow.
  int maxClones = 1;
  // The number of errors to emit. Set to -1 to emit all errors. Cannot be 0.
  int errorLimit = 5;
};

// Infers the mesh assignments of non-mpmd ops that are not nested within a
// fragment op. This uses an analysis to determine the mesh assignments. See
// comments in the implementation for more details.
//
// These passes work to infer the meshes of all unassigned ops and func inputs
// and outputs, with the restriction that we do not create any new transfers
// except on func inputs.
//
// Between the analysis and rewrite, there shouldn't be any passes that do DCE.
// E.g. no use of the greedy rewriter.
void addInferMeshPipeline(
    OpPassManager& pm,
    SmallVector<InputOutputEquishardingConstraint> inputOutputConstraints,
    InferMeshOptions options = {});

// Register the `-mpmd-infer-mesh-pipeline`.
void registerInferMeshPipeline();

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_INFER_MESH_ASSIGNMENT_H_
