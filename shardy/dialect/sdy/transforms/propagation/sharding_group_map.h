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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_GROUP_MAP_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_GROUP_MAP_H_

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace sdy {

class ShardingGroupMap {
 public:
  ShardingGroupMap(ModuleOp moduleOp);

  // For each sharding group, sync the shardings of all its members by applying
  // the common sharding of members with an existing non-replicated sharding.
  //
  // If there is a conflict in the initial shardings of members within a
  // sharding group, then we insert an open sharding constraint on all members
  // in the group, which replace the members themselves. This is to ensure that
  // the group is still valid after propagation.
  void syncGroupMemberShardings(ModuleOp module);

  // Returns the set of Values which are in the same sharding group as `value`
  // (including `value`) or an empty range if none exist.
  ValueRange getGroupMembers(const Value& value) const;

 private:
  SmallVector<SmallVector<Value>> shardingGroupToValues;
  llvm::SmallDenseMap<Value, int64_t> valueToShardingGroup;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_GROUP_MAP_H_
