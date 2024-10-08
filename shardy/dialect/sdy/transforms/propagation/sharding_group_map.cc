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

#include <cassert>
#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

namespace {

using ShardingGroupToValues = SmallVector<SmallVector<Value>>;
using ValueToShardingGroup = DenseMap<Value, int64_t>;

}  // namespace

LogicalResult ShardingGroupMap::buildFromModule(ModuleOp moduleOp) {
  DenseMap<int64_t, SmallVector<Value>> groupMap;
  WalkResult result = moduleOp.walk([&](ShardingGroupOp op) {
    // Assert that each value corresponds to only one sharding group id
    // after group canonicalization.
    if (valueToShardingGroup.contains(op.getInput())) {
      assert(valueToShardingGroup.at(op.getInput()) == op.getGroupId() &&
             "value can only map to one sharding group id after "
             "canonicalization.");
      return WalkResult::interrupt();
    }
    valueToShardingGroup[op.getInput()] = op.getGroupId();
    groupMap[op.getGroupId()].push_back(op.getInput());
    return WalkResult::advance();
  });

  for (int i = 0; i < groupMap.size(); ++i) {
    shardingGroupToValues.push_back(groupMap[i]);
  }
  return failure(result.wasInterrupted());
}

int64_t ShardingGroupMap::getShardingGroupId(Value value) const {
  auto it = valueToShardingGroup.find(value);
  if (it == valueToShardingGroup.end()) {
    return -1;
  }
  return it->getSecond();
}

SmallVector<Value> ShardingGroupMap::getValuesForGroup(
    int64_t shardingGroupId) const {
  if (shardingGroupId < 0 || shardingGroupId >= shardingGroupToValues.size()) {
    return {};
  }
  return shardingGroupToValues[shardingGroupId];
}

}  // namespace sdy
}  // namespace mlir
