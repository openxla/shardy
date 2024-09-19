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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace sdy {

class ShardingGroupMap {
 public:
  ShardingGroupMap() = default;
  LogicalResult buildFromModule(ModuleOp moduleOp);
  int64_t getShardingGroupId(Value value) const;
  SmallVector<Value> getValuesForGroup(int64_t shardingGroupId) const;

 private:
  SmallVector<SmallVector<Value>> shardingGroupToValues;
  DenseMap<Value, int64_t> valueToShardingGroup;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_GROUP_MAP_H_
