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
#include <functional>
#include <iterator>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
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
                    return !mesh.getName().ends_with("#cpu");
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

void AddControlDependency(FragmentOp fragment1, FragmentOp fragment2,
                          DenseMap<FragmentOp, int>& ctrl_dependency_counter) {
  // We add a new operand at the end.
  int operand_index = fragment2.getNumOperands();
  fragment2->insertOperands(operand_index, {fragment1->getResult(0)});
  ctrl_dependency_counter[fragment2] += 1;
}

void RemoveAllControlDependencies(
    DenseMap<FragmentOp, int>& ctrl_dependency_counter) {
  for (auto& [fragment, counter] : ctrl_dependency_counter) {
    const int start_index = fragment->getNumOperands() - counter;
    fragment->eraseOperands(start_index, counter);
  }
}

namespace {

// Callback type for `VisitOpUseTree`.
using PostActionCallBack = std::function<bool(Operation*)>;

// Visits all users in a depth-first way, starting from current, with the
// constraint that the traversal remains within the same block as the
// `barrier_op` and never visits a node _after_ the `barrier_op`. After all the
// users have been recursively visited it invokes the `action` callback. The
// traversal terminates early if one of these callbacks returns `false`.
bool VisitOpUseTree(Operation* current, Operation* barrier_op,
                    DenseSet<Operation*>& visited, PostActionCallBack action) {
  // Invariant: we will always have started with a `current` op at the same
  // block as `barrier` op; hence it is always possible to trace all recursive
  // users of `current` to ancestors in the same block as `barrier_op`.
  current = GetAncestorInBlock(barrier_op->getBlock(), current);
  // Done traversing if we are already visited or are beyond the barrier.
  if (barrier_op->isBeforeInBlock(current) || visited.contains(current)) {
    return true;
  }

  for (Value result : current->getResults()) {
    bool any_failure = llvm::any_of(result.getUsers(), [&](Operation* user) {
      return !VisitOpUseTree(user, barrier_op, visited, action);
    });
    if (any_failure) return false;
  }
  visited.insert(current);
  return action(current);
}

}  // namespace

bool TargetDependsOnSourceOp(Operation* src_op, Operation* tgt_op) {
  SDY_CHECK(src_op->getBlock() == tgt_op->getBlock());
  DenseSet<Operation*> visited;
  return !VisitOpUseTree(src_op, tgt_op, visited,
                         [&](Operation* op) { return op != tgt_op; });
}

}  // namespace mlir::mpmd
