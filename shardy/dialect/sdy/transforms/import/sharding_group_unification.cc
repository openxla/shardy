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
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SHARDINGGROUPUNIFICATIONPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using llvm::DenseMap;
using llvm::EquivalenceClasses;
using llvm::SmallDenseMap;
using llvm::SmallVector;

struct ShardingGroupUnificationPass
    : public impl::ShardingGroupUnificationPassBase<
          ShardingGroupUnificationPass> {
  using ShardingGroupUnificationPassBase::ShardingGroupUnificationPassBase;

  void runOnOperation() final {
    // Extract the sharding group ids and tensor -> {group_id} mapping from the
    // high level module, and initialize the equivalence classes for the group
    // ids present.
    DenseMap<Value, SmallVector<ShardingGroupOp>> tensorToGroups;
    ModuleOp module = getOperation();
    module.walk([&](ShardingGroupOp op) {
      tensorToGroups[op.getInput()].push_back(op);
    });
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
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
