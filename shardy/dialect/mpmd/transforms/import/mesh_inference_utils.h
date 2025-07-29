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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_INFERENCE_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_INFERENCE_UTILS_H_

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"

namespace mlir::mpmd {

// Attribute name representing a set of meshes that a meshless op is
// transitively used in. This propagates backward from users to the op by union.
// It tells us where the op must live: an op (or a copy of) must live on all
// of its uses since we don't allow the introduction of transfers except on
// function inputs and so the entire chain of ops must be in a single mesh (but
// the chain could be duplicated across meshes).
//
// If the attribute is not present on an unassigned op, then the op is assumed
// to have use_set = {} (i.e. the set is empty).
//
// See initialization passes below for details on initialization.
inline constexpr StringRef kMpmdUseSet = "mpmd.use_set";

// Attribute name representing a set of meshes that a meshless op can
// exist on without introducing transfer ops. For unassign ops,
// this is where the op/result actually lives. However, child ops are not
// assigned a mesh yet, and so although the child op could be placed on a mesh,
// it may not be. For example, in
//
// y = unassign x
// z = transfer x m1 -> m2
// yy = add y, y
//
// Then `add y, y` can live on both m1 and m2. But where it eventually lives on
// depends on the use_set. Before assignment, yy doesn't live on m1 or m2 yet –
// it could live on one of them, or both of them.
//
// This propagates forward from operands to the op by intersection, since an op
// and its operands and results must live on the same mesh.
//
// If the attribute is not present on an unassigned op, then the op is assumed
// to have src_set = {all meshes}.
//
// See initialization passes below for details on initialization.
inline constexpr StringRef kMpmdSrcSet = "mpmd.src_set";

// Temporary attribute name to store whether an op can be converted to a
// mpmd.reduce from an analysis of the user-added mpmd.reduce<none>.
inline constexpr StringRef kCanConvertToReduce = "mpmd.can_convert_to_reduce";

// Temporary attribute name to store the inferred mpmd reduce info.
inline constexpr StringRef kMpmdReduceAnnotation = "mpmd.reduce";

// Returns true if the given `op` needs to be assigned to a mesh.
bool IsMeshlessOp(Operation* op);

// Returns the use_set. We return an empty use_set if it is missing, since they
// are interchangeable.
MeshesWithOrigins GetUseSet(Operation* op);
MeshesWithOrigins GetArgUseSet(func::FuncOp func, int arg);
MeshesWithOrigins GetResUseSet(func::FuncOp func, int res);
MeshesWithOrigins GetUseSet(ForOp for_op, int arg_number);
// Returns the use_set as a set of mesh names.
llvm::SetVector<StringRef> GetUseMeshes(Operation* op);

// Set the use_set attribute if the set is not empty. This is because the
// absence of the attribute is the same as the empty list, and we do not want to
// clutter the IR.
void SetUseSet(Operation* op, MeshesWithOrigins use_set, OpBuilder& builder);
void SetArgUseSet(func::FuncOp func, int arg, MeshesWithOrigins use_set,
                  OpBuilder& builder);
void SetResUseSet(func::FuncOp func, int res, MeshesWithOrigins use_set,
                  OpBuilder& builder);
void SetUseSet(ForOp for_op, int arg_number, MeshesWithOrigins use_set,
               OpBuilder& builder);

// Returns the src_set of an op or func arg.
MeshesWithOrigins GetSrcSet(Operation* op);
MeshesWithOrigins GetSrcSet(func::FuncOp func, int arg_number);
// Returns the src_set as a set of mesh names.
std::optional<llvm::SetVector<StringRef>> GetSrcMeshes(Operation* op);
std::optional<llvm::SetVector<StringRef>> GetSrcMeshes(func::FuncOp func,
                                                       int arg_number);
MeshesWithOrigins GetSrcSet(ForOp for_op, int arg_number);

// Returns the src_set of an operand, passing through data flow ops as required.
MeshesWithOrigins GetSrcSet(OpOperand& op_operand);
std::optional<llvm::SetVector<StringRef>> GetSrcMeshes(OpOperand& op_operand);

// Set the src_set attribute. Unlike use_set, we set it even if it is empty,
// because the empty src_set means that the op cannot live anywhere. Whereas
// absence of the attribute means that the op can live on any mesh.
void SetSrcSet(Operation* op, MeshesWithOrigins src_set, OpBuilder& builder);
void SetSrcSet(func::FuncOp func, int arg_number, MeshesWithOrigins src_set,
               OpBuilder& builder);
void SetSrcSet(ForOp for_op, int arg_number, MeshesWithOrigins src_set,
               OpBuilder& builder);

// Removes the use_set attributes. Returns true if the attribute was removed.
bool ClearUseSet(Operation* op);
bool ClearUseSet(func::FuncOp func);

// Removes the src_set and use_set attributes. Returns true if any attribute
// was removed.
bool ClearUseSetAndSrcSet(Operation* op);
bool ClearUseSetAndSrcSet(func::FuncOp func);
bool ClearUseSetAndSrcSet(ForOp for_op);

// Removes the can_convert_to_reduce attribute.
void ClearCanConvertAttr(func::FuncOp func);

// Gets transitive uses of ops by combining the use_sets of users, updating
// `base_set` in place. This assumes that all the users have the relevant
// use_set data populated. It passes through data flow ops, treating them as
// "transparent".
void UpdateTransitiveUses(Operation* op, MeshesWithOrigins& base_set);
void UpdateTransitiveUses(Value value, MeshesWithOrigins& base_set);

// Returns if the op is a terminal node in the mesh analysis. I.e. it blocks
// propagation of use/src analysis. The only such ops are: the mpmd.reduce and
// mpmd.broadcast ops.
bool IsTerminalNodeInAnalysis(Operation* op);

inline bool IsMpmdReduceAnnotated(Operation* op) {
  return op->hasAttr(kMpmdReduceAnnotation);
}

inline bool CanConvertToReduce(Operation* op) {
  return op->hasAttr(kCanConvertToReduce);
}

// Returns true if the given `call_op` is in a call chain with itself. E.g.
// we have a chain of calls like:
//
// %v = call @f(%v0)
// %w = call @f(%v)
bool IsCallOpInCallChain(CallOp call_op);

// Returns true if the use_set of the callee func has been populated.
// This checks if any of the callee func's args have the use_set attribute
// populated as a proxy.
bool CallOpHasUseSetPopulated(CallOp call_op);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_INFERENCE_UTILS_H_
