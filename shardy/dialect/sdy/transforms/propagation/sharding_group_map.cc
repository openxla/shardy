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
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

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
