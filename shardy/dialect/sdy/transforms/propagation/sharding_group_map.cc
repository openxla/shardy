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

#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

ShardingGroupMap::ShardingGroupMap(ModuleOp moduleOp) {
  moduleOp.walk([&](ShardingGroupOp op) {
    // After canonicalization all group ids will take distinct values in the
    // range 0,1,...,N. Because of this we can directly index these group ids
    // into the shardingGroupToValues vector (resizing when necessary).
    shardingGroupToValues.resize(
        std::max(op.getGroupId() + 1,
                 static_cast<uint64_t>(shardingGroupToValues.size())));
    // Each value can only map to one sharding group id after
    // canonicalization.
    auto [it, inserted] =
        valueToShardingGroup.try_emplace(op.getInput(), op.getGroupId());
    if (!inserted && it->getSecond() != op.getGroupId()) {
      llvm::report_fatal_error(
          "Value can only map to one sharding group id after import.");
    }
    shardingGroupToValues[op.getGroupId()].push_back(op.getInput());
  });
}

namespace {

struct CommonSharding {
  TensorShardingAttr sharding;
  bool hasConflict;
};

// Find the common sharding for the given sharding group. Also returns a bool
// indicating whether there is conflict between some shardings in the group
// If they do, the returned sharding is the sharding of the first value in the
// group.
CommonSharding findCommonSharding(int64_t groupId, ValueRange groupMembers) {
  if (groupMembers.size() == 1) {
    return {getSharding(groupMembers.front()), false};
  }
  Value firstMember = groupMembers.front();
  TensorShardingAttr sharding;
  for (Value member : groupMembers) {
    TensorShardingAttr candidateSharding = getSharding(member);
    if (!candidateSharding || candidateSharding.isFullyReplicatedAndOpen()) {
      continue;
    }
    if (!sharding) {
      sharding = candidateSharding;
    } else if (candidateSharding != sharding) {
      // TODO(bartchr): Revisit using jax.shard_alike as the example once others
      // like PyTorch use Shardy.
      getOwningOp(firstMember)
              ->emitWarning(
                  "The initial operand shardings on the sharding groups of "
                  "groupID: ")
          << groupId << " do not match. Inserting an open "
          << "sharding constraint to all constrained values. "
          << "This can be caused when shardings from different values are "
          << "grouped (e.g. from jax.shard_alike) but have separate "
          << "inconsistent sharding constraints on them.";
      return {sharding, true};
    }
  }

  return {sharding, false};
}

}  // namespace

void ShardingGroupMap::syncGroupMemberShardings(ModuleOp module) {
  for (auto [groupId, groupMembers] : llvm::enumerate(shardingGroupToValues)) {
    auto [sharding, hasConflict] = findCommonSharding(groupId, groupMembers);
    if (!sharding) {
      continue;
    }
    if (!hasConflict) {
      for (Value member : groupMembers) {
        setSharding(member, sharding);
      }
      continue;
    }
    // NOTE: Arbitrarily use the mesh name of `sharding`. This would be an
    // issue, before we support propagating through different meshes, if the
    // existing mesh of the member is different. It's fine to not error since
    // this is a really rare case.
    for (Value& member : groupMembers) {
      IRRewriter rewriter(module.getContext());
      rewriter.setInsertionPointAfterValue(member);
      auto openShardingConstraint = ShardingConstraintOp::create(
          rewriter, member.getLoc(), member,
          TensorShardingAttr::getFullyOpenLike(sharding));
      rewriter.replaceAllUsesExcept(member, openShardingConstraint,
                                    openShardingConstraint);
      // Replace member.
      valueToShardingGroup.erase(member);
      member = openShardingConstraint;
      valueToShardingGroup.try_emplace(member, groupId);
    }
  }
}

ValueRange ShardingGroupMap::getGroupMembers(const Value& value) const {
  if (auto it = valueToShardingGroup.find(value);
      it != valueToShardingGroup.end()) {
    int64_t shardingGroupId = it->getSecond();
    return shardingGroupToValues[shardingGroupId];
  }
  return {};
}

}  // namespace sdy
}  // namespace mlir
