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

#include <grp.h>

#include <cstdint>
#include <memory>  // IWYU pragma: keep

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_GROUPCANONICALIZATIONPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using llvm::DenseMap;
using llvm::DenseSet;
using llvm::EquivalenceClasses;
using llvm::SmallVector;
using ShardGroupId = int64_t;

struct GroupCanonicalizationPass
    : public impl::GroupCanonicalizationPassBase<GroupCanonicalizationPass> {
  using GroupCanonicalizationPassBase::GroupCanonicalizationPassBase;

  void runOnOperation() final {
    // Extract the sharding group ids and tensor -> {group_id} mapping from the
    // high level module, and initialize the equivalence classes for the group
    // ids present.
    DenseSet<ShardGroupId> uniqueGroupIds;
    DenseMap<Value, SmallVector<ShardGroupId>> tensorToGroups;
    ModuleOp module = getOperation();
    module.walk([&](ShardingGroupOp op) {
      ShardGroupId gid = op.getGroupId();
      Value inputTensor = op.getInput();
      uniqueGroupIds.insert(gid);
      tensorToGroups[inputTensor].push_back(gid);
    });
    EquivalenceClasses<ShardGroupId> ec;
    for (ShardGroupId gid : uniqueGroupIds) {
      ec.insert(gid);
    }

    // Merge the equivalence classes of group ids which had the same tensors
    // within them. (unionSets uses the default comparator and will consider the
    // minimal group_id as the representative element of the equivalence class).
    for (const auto& [_, groupsForTensor] : tensorToGroups) {
      if (groupsForTensor.empty()) {
        continue;
      }
      ShardGroupId canonicalId = groupsForTensor.front();
      SmallVector<ShardGroupId> groupsToMerge(groupsForTensor.begin() + 1,
                                              groupsForTensor.end());
      for (ShardGroupId groupId : groupsToMerge) {
        ec.unionSets(canonicalId, groupId);
      }
    }

    // Rewalk the graph to replace group_ids with their canonical id.
    module.walk([&](ShardingGroupOp op) {
      ShardGroupId canonicalId = ec.getLeaderValue(op.getGroupId());
      op.setGroupId(canonicalId);
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
