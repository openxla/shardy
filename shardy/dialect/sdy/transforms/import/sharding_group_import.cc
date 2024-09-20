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

#include <cstdint>
#include <memory>  // IWYU pragma: keep

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SHARDINGGROUPIMPORTPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using llvm::DenseMap;
using llvm::EquivalenceClasses;
using llvm::SmallDenseMap;
using llvm::SmallVector;

using ValueToShardingGroup =
    llvm::DenseMap<Value, llvm::SmallVector<ShardingGroupOp>>;

void unifyShardingGroups(ValueToShardingGroup& tensorToGroups) {
  if (tensorToGroups.empty()) {
    return;
  }
  // Merge the equivalence classes of group ids which had the same tensors
  // within them. (unionSets uses the default comparator and will consider the
  // minimum group_id as the representative element of the equivalence class).
  EquivalenceClasses<int64_t> shardingGroupEquivalences;
  for (auto& [_, groupsForTensor] : tensorToGroups) {
    const int64_t canonicalId = groupsForTensor.front().getGroupId();
    for (ShardingGroupOp group : groupsForTensor) {
      shardingGroupEquivalences.unionSets(canonicalId, group.getGroupId());
    }
  }

  // After merging groups we reindex the group IDs so that they take values
  // from the set {0,1,...,N-1} (N is the number of equivalence classes).
  // The leader element of each equivalent class corresponds to the minimum
  // group_id, so by looping over the group leaders in order their reindexed
  // ids can be set to maintain the same relative ordering.
  int64_t reindexId = 0;
  SmallDenseMap<int64_t, int64_t> reindexMap;
  for (const auto& group : shardingGroupEquivalences) {
    if (group.isLeader()) {
      reindexMap[group.getData()] = reindexId++;
    }
  }

  // Update the graph to replace group_ids with their canonical id.
  for (auto& [_, groupsForTensor] : tensorToGroups) {
    for (ShardingGroupOp op : groupsForTensor) {
      op.setGroupId(reindexMap[shardingGroupEquivalences.getLeaderValue(
          op.getGroupId())]);
    }
  }
}

LogicalResult buildShardingGroupMappingAndValidateGroups(
    ModuleOp module, ValueToShardingGroup& tensorToGroups) {
  // Map to hold validation info for shard groups within manual computations.
  DenseMap<int64_t, ManualComputationOp> groupToManualComp;
  DenseMap<int64_t, ArrayRef<int64_t>> groupToTensorShape;

  // While walking the graph we simultaneously build up the tensorToGroups
  // mapping (which will be used for unification) while also validating the
  // structure of shard groups meets expectations
  WalkResult result = module.walk([&](ShardingGroupOp op) {
    tensorToGroups[op.getInput()].push_back(op);

    // All values in a sharding group should have either:
    // 1) No manual computation op parent
    // 2) The same manual computation op parent.
    // If a group has no manual computation op parent, 'groupToManualComp'
    // will map it to nullptr and ensure all other values in that group are
    // also mapped to nullptr.
    auto parent = op->getParentOfType<ManualComputationOp>();
    int64_t groupId = op.getGroupId();

    auto [it, inserted] = groupToManualComp.try_emplace(groupId, parent);
    if (!inserted && it->second != parent) {
      op.emitError(
          "ShardingGroupOps values cannot cross ManualComputationOp "
          "boundaries for groupId: ")
          << groupId;
      return WalkResult::interrupt();
    }

    // All values in a sharding group should have the same shape.
    ArrayRef<int64_t> tensorShape = getTensorShape(op.getInput());
    auto [itTensorShape, tensorShapeInserted] =
        groupToTensorShape.try_emplace(groupId, tensorShape);
    if (!tensorShapeInserted && itTensorShape->getSecond() != tensorShape) {
      op.emitError(
          "ShardingGroupOps values must have the same shape for groupId: ")
          << groupId;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

struct ShardingGroupImportPass
    : public impl::ShardingGroupImportPassBase<ShardingGroupImportPass> {
  using ShardingGroupImportPassBase::ShardingGroupImportPassBase;

  void runOnOperation() final {
    // Extract the sharding group ids and tensor -> {group_id} mapping from the
    // high level module and validate any sharding group constrainst are met.
    ValueToShardingGroup tensorToGroups;
    if (failed(buildShardingGroupMappingAndValidateGroups(getOperation(),
                                                          tensorToGroups))) {
      signalPassFailure();
    }

    unifyShardingGroups(tensorToGroups);
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
