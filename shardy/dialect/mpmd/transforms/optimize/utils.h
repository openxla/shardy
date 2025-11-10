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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_UTILS_H_

#include <optional>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"

namespace mlir::mpmd {

// Returns true if fragment is a forward fragment, i.e., if the transpose count
// is 0.
bool IsForwardFragment(FragmentOp fragment);

// Returns true if fragment is a backward fragment, i.e., if the transpose count
// is 1.
bool IsBackwardFragment(FragmentOp fragment);

// Given two fragments, returns true if the former can be rematerialized for
// consumption of the latter, i.e., we check that:
// - forward_fragment is a forward fragment
// - backward_fragment is a backward fragment.
// - Their call counters match.
// - Their stage ids match, when defined.
bool CanRemat(FragmentOp forward_fragment, FragmentOp backward_fragment);

// Returns the meshes in the topology that are schedulable for pipeline
// computations. I.e. they are not CPU meshes.
SmallVector<NamedMeshAttr> GetSchedulableMeshes(func::FuncOp func);

// Returns the number of meshes in the pipeline.
// TODO(jupvfranco): This code is assuming that every mesh in the topology is a
// pipeline stage. Generalize this.
int GetNumMeshes(Operation* op);

// Returns the index of the mesh of `fragment` in the pipeline.
// TODO(jupvfranco): This code is assuming that the meshes appear in the order
// of stages. Generalize this once we have stages.
int GetMeshIndex(FragmentOp fragment);

// Checks if a fragment is a scheduling unit, i.e., it is a user fragment,
// it has a call_counter and a single transpose_count which is 0 or 1.
bool IsSchedulingUnit(FragmentOp fragment);

// Does `tgt_op` have (conservatively) any dataflow dependency on `src_op`?
// Precondition: `tgt_op` and `src_op` must be in the same block.

// Gets the dataflow dependency path from `src_op` to `tgt_op` if it exists.
//
// A dependency exists if `tgt_op` is in the use-def chain of `src_op` and does
// not appear before `src_op` within the same block.
//
// Returns the dependency path as a vector of operations from `src_op` to
// `tgt_op` if a dependency exists, otherwise returns `std::nullopt`.
//
// Precondition: `src_op` and `tgt_op` must be in the same block.
std::optional<SmallVector<Operation*>> GetDependencyPath(Operation* src_op,
                                                         Operation* tgt_op);

// Adds a control dependency in the graph so that `fragment2` depends on
// `fragment1`. This is done by inserting fragment1's result as a operand of
// fragment2. We call these temporary operands "control operands".
//
// Control operands are tracked via the kControlOperandStartIdxAttrName
// attribute, which marks the index where control operands begin. This allows
// control dependencies to persist across multiple passes until explicitly
// removed by RemoveAllControlDependencies().
//
// NOTE: This creates a temporarily ill-formed FragmentOp since control operands
// are not actual data dependencies. The fragment must not be verified or used
// in operations that expect well-formed fragments until the control
// dependencies are removed.
void AddControlDependency(FragmentOp fragment1, FragmentOp fragment2);

// Removes all control dependencies from fragments in the function, restoring
// them to a well-formed state.
//
// Control dependencies are identified by the kControlOperandStartIdxAttrName
// attribute. All operands from that index onwards are removed, and the
// attribute itself is deleted.
//
// This should be called after all passes use topological sorting have completed
// (after which the control dependencies no longer have any use).
void RemoveAllControlDependencies(func::FuncOp func_op);

// Formats a detailed warning message for a scheduling conflict.
std::string FormatConflictWarning(const FragmentInfo& predecessor_info,
                                  const FragmentInfo& successor_info,
                                  const SmallVector<Operation*>& conflict_path);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_UTILS_H_
