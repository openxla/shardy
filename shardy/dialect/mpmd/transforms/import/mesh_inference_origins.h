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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_INFERENCE_ORIGINS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_INFERENCE_ORIGINS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

// The following strings are the origin labels for AssignOps and UnassignOps.
// They are used to indicate the source of the mesh assignment, and may be used
// for merging.

// Mesh assignment originates from the user. I.e. input_mesh_assignment or
// in_sharding.
inline constexpr StringRef kUserInputOrigin = "user_in";
// Mesh assignment originates from the user. I.e. input_mesh_assignment or
// in_sharding.inline constexpr StringRef kUserOutputOrigin = "user_out";
inline constexpr StringRef kUserOutputOrigin = "user_out";
// Mesh assignment originates from mesh inference where we assign the input
// using our analysis.
inline constexpr StringRef kInferredInputOrigin = "inferred_in";
// Mesh assignment originates from mesh inference where we assign the output
// using our analysis.
inline constexpr StringRef kInferredOutputOrigin = "inferred_out";
// Mesh assignment originates from mesh inference where we assign an unused op
// using our analysis.
inline constexpr StringRef kInferredUnusedOrigin = "inferred_unused";
// Mesh assignment originates from mesh inference where we assign an unused
// callee's input/output using our analysis.
inline constexpr StringRef kInferredUnusedCalleeInOrigin =
    "inferred_unused_callee_in";
inline constexpr StringRef kInferredUnusedCalleeOutOrigin =
    "inferred_unused_callee_out";
// Mesh assignment originates from input/output constraints.
inline constexpr StringRef kIoConstraintInputOrigin = "io_constraint_in";
// Mesh assignment originates from input/output constraints.
inline constexpr StringRef kIoConstraintOutputOrigin = "io_constraint_out";

inline constexpr StringRef kTransferOrigin = "transfer";

// Mesh assignment originates from broadcasting inputs.
inline constexpr llvm::StringRef kBroadcastInputOrigin = "broadcast_in";
// Mesh assignment originates from an mpmd.broadcast op.
inline constexpr StringRef kMpmdBroadcastOrigin = "broadcast";
// Mesh assignment originates from an mpmd.reduce op.
inline constexpr StringRef kMpmdReduceOrigin = "reduce";

// Origin string for terminal nodes (e.g. mpmd.broadcast and mpmd.reduce).
StringRef TerminalNodesOrigin(Operation* op);

MeshWithOriginsAttr UnusedCalleeInputMeshWithOrigin(MLIRContext* context,
                                                    StringRef mesh_name);
MeshWithOriginsAttr UnusedCalleeOutputMeshWithOrigin(MLIRContext* context,
                                                     StringRef mesh_name);

MeshWithOriginsAttr TransferMeshWithOrigin(TransferOp transfer_op);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_INFERENCE_ORIGINS_H_
