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

#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

namespace {

bool CallCounterMatchesForRemat(FragmentOp forward_fragment,
                                FragmentOp backward_fragment) {
  std::optional<int64_t> forward_call_counter =
      TryToFindCallCounter(forward_fragment);
  std::optional<int64_t> backward_call_counter =
      TryToFindCallCounter(backward_fragment);
  return (forward_call_counter.has_value() &&
          backward_call_counter.has_value() &&
          *forward_call_counter == *backward_call_counter);
}

}  // namespace

bool IsForwardFragment(FragmentOp fragment) {
  std::optional<int64_t> transpose_count =
      TryToFindSingleTransposeCount(fragment);
  return transpose_count.has_value() && *transpose_count == 0;
}

bool IsBackwardFragment(FragmentOp fragment) {
  std::optional<int64_t> transpose_count =
      TryToFindSingleTransposeCount(fragment);
  // We only consider backward fragments that have transpose_count 1, which
  // means there is exactly one Jax AD transpose. If the value is
  // greater than 1, it means there are multiple jax.grad and it is unclear what
  // remat means in this case.
  return transpose_count.has_value() && *transpose_count == 1;
}

bool CanRemat(FragmentOp forward_fragment, FragmentOp backward_fragment) {
  return IsForwardFragment(forward_fragment) &&
         IsBackwardFragment(backward_fragment) &&
         CallCounterMatchesForRemat(forward_fragment, backward_fragment) &&
         !IsExecutedImmediatelyAfter(forward_fragment, backward_fragment) &&
         forward_fragment.getStageIdAttr() ==
             backward_fragment.getStageIdAttr();
}

SmallVector<mpmd::NamedMeshAttr> GetSchedulableMeshes(func::FuncOp func) {
  auto all_meshes = mpmd::GetTopologyMeshes(func);
  SmallVector<mpmd::NamedMeshAttr> hbm_meshes;
  llvm::copy_if(all_meshes, std::back_inserter(hbm_meshes),
                [](const mpmd::NamedMeshAttr& mesh) {
                  return !mesh.getName().ends_with(mpmd::kCpuMeshSuffix);
                });
  return hbm_meshes;
}

// Returns the number of meshes in the pipeline.
// TODO(jupvfranco): This code is assuming that every mesh in the topology is a
// pipeline stage. Generalize this.
int GetNumMeshes(Operation* op) {
  return GetSchedulableMeshes(op->getParentOfType<func::FuncOp>()).size();
}

// Returns the index of the mesh of `fragment` in the pipeline.
// TODO(jupvfranco): This code is assuming that the meshes appear in the order
// of stages. Generalize this once we have stages.
int GetMeshIndex(FragmentOp fragment) {
  for (auto [index, mesh] : llvm::enumerate(
           GetSchedulableMeshes(fragment->getParentOfType<func::FuncOp>()))) {
    if (mesh.getName() == fragment.getMeshName()) {
      return index;
    }
  }
  // At this point we are guaranteed that every fragment's mesh is in the
  // topology.
  SDY_CHECK(false) << "Mesh not found: " << fragment.getMeshName().str();
}

// Checks if a fragment is a scheduling unit, i.e., it is a user fragment,
// it has a call_counter and a single transpose_count which is 0 or 1.
bool IsSchedulingUnit(FragmentOp fragment) {
  if (!fragment.isUserFragment()) {
    return false;
  }

  if (!TryToFindCallCounter(fragment).has_value()) return false;

  if (auto transpose_count = TryToFindSingleTransposeCount(fragment);
      transpose_count.has_value()) {
    return *transpose_count == 0 || *transpose_count == 1;
  }

  return false;
}

void AddControlDependency(FragmentOp fragment1, FragmentOp fragment2) {
  // Store control operand start index (only if not already stored)
  if (!fragment2->hasAttr(kControlOperandStartIdxAttrName)) {
    MLIRContext* context = fragment2->getContext();
    int64_t control_operand_start_index = fragment2.getNumOperands();
    fragment2->setAttr(kControlOperandStartIdxAttrName,
                       IntegerAttr::get(IntegerType::get(context, 64),
                                        control_operand_start_index));
  }

  // Add control operand at the end
  int operand_index = fragment2.getNumOperands();
  fragment2->insertOperands(operand_index, {fragment1->getResult(0)});
}

void RemoveAllControlDependencies(func::FuncOp func_op) {
  func_op.walk([](FragmentOp fragment) {
    if (auto attr = fragment->getAttrOfType<IntegerAttr>(
            kControlOperandStartIdxAttrName)) {
      int64_t control_operand_start_index = attr.getInt();
      int64_t current_operand_count = fragment->getNumOperands();
      int64_t control_operand_count =
          current_operand_count - control_operand_start_index;

      SDY_CHECK(control_operand_count > 0)
          << "kControlOperandStartIdxAttrName present but no control operands "
             "found. This should not happen, contact MPMD team for support. "
             "current_operand_count: "
          << current_operand_count
          << ", control_operand_start_index: " << control_operand_start_index;

      // Remove all control operands (from control_operand_start_index to end)
      fragment->eraseOperands(control_operand_start_index,
                              control_operand_count);

      // Remove the attribute after cleanup
      fragment->removeAttr(kControlOperandStartIdxAttrName);
    }
  });
}

namespace {

// Visits all users in a depth-first, pre-order way, starting from current,
// with the constraint that the traversal remains within the same block as the
// `target` and never visits a node after the `target`. Pre-order is used to
// exit faster and so that each operation is added to the path before exploring
// its children, building the dependency path incrementally. Returns true if the
// target is found, false otherwise. When target is found, the path parameter
// contains the complete dependency path.
bool TraverseToTarget(Operation* current, Operation* target,
                      DenseSet<Operation*>& visited,
                      SmallVector<Operation*>& path) {
  // Invariant: we will always have started with a `current` op at the same
  // block as `target`; hence it is always possible to trace all recursive
  // users of `current` to ancestors in the same block as `target`.
  current = GetAncestorInBlock(target->getBlock(), current);
  auto [_, was_inserted] = visited.insert(current);
  // Done traversing if we have already visited or are beyond the target.
  if (target->isBeforeInBlock(current) || !was_inserted) {
    return false;
  }

  path.push_back(current);

  // Check if we found the target.
  if (current == target) {
    return true;
  }

  // Explore all children
  for (Value result : current->getResults()) {
    for (Operation* user : result.getUsers()) {
      if (TraverseToTarget(user, target, visited, path)) {
        return true;
      }
    }
  }

  // Target not found in any subtree, so remove current from path
  path.pop_back();
  return false;
}

}  // namespace

std::optional<SmallVector<Operation*>> GetDependencyPath(Operation* src_op,
                                                         Operation* tgt_op) {
  SDY_CHECK(src_op->getBlock() == tgt_op->getBlock());
  DenseSet<Operation*> visited;
  SmallVector<Operation*> path;

  if (TraverseToTarget(src_op, tgt_op, visited, path)) {
    return path;
  }
  return std::nullopt;
}

std::string FormatConflictWarning(
    const FragmentInfo& predecessor_info, const FragmentInfo& successor_info,
    const SmallVector<Operation*>& conflict_path) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "Scheduling rule conflict: rule specifies that\n"
     << "  " << llvm::to_string(predecessor_info) << "\n"
     << "must be scheduled before\n"
     << "  " << llvm::to_string(successor_info) << "\n"
     << "but a dataflow dependency requires the opposite order.\n"
     << "Conflicting dependency path:\n";
  for (auto [i, op] : llvm::enumerate(conflict_path)) {
    os << "  [" << i << "] ";
    if (auto fragment = dyn_cast<FragmentOp>(op)) {
      os << llvm::to_string(GetFragmentInfo(fragment));
    } else {
      os << op->getName().getStringRef();
    }
    os << "\n";
  }
  return message;
}

}  // namespace mlir::mpmd
