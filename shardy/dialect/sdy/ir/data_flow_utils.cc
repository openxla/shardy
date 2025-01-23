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

#include "shardy/dialect/sdy/ir/data_flow_utils.h"

#include <cassert>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

// Returns the owning op if it is a `ShardableDataFlowOpInterface`, otherwise
// returns nullptr.
ShardableDataFlowOpInterface dynCastOwningShardableDataFlowOp(Value value) {
  return dyn_cast<ShardableDataFlowOpInterface>(getOwningOp(value));
}

// Returns the owning `ShardableDataFlowOpInterface` (assuming it exists).
ShardableDataFlowOpInterface castOwningShardableDataFlowOp(Value value) {
  ShardableDataFlowOpInterface shardableDataFlowOp =
      dynCastOwningShardableDataFlowOp(value);
  assert(shardableDataFlowOp);
  return shardableDataFlowOp;
}

Value getDataFlowEdgeOwner(Value target) {
  if (ShardableDataFlowOpInterface shardableDataFlowOp =
          dynCastOwningShardableDataFlowOp(target)) {
    return shardableDataFlowOp.getEdgeOwnerFromTarget(target);
  }
  return nullptr;
}

Value getDataFlowEdgeOwner(OpOperand& source) {
  Operation* op = source.getOwner();
  op = op->hasTrait<OpTrait::IsTerminator>() ? op->getParentOp() : op;
  if (auto shardableDataFlowOp = dyn_cast<ShardableDataFlowOpInterface>(op)) {
    return shardableDataFlowOp.getEdgeOwnerFromSource(source);
  }
  return nullptr;
}

}  // namespace

DataFlowEdgeOp getDataFlowEdge(Value target) {
  return DataFlowEdgeOp::getDataFlowEdgeUser(getDataFlowEdgeOwner(target));
}

DataFlowEdgeOp getDataFlowEdge(OpOperand& source) {
  return DataFlowEdgeOp::getDataFlowEdgeUser(getDataFlowEdgeOwner(source));
}

TensorShardingAttr transformTargetSharding(
    DataFlowEdgeOp dataFlowEdge, TensorShardingAttr sharding,
    DataFlowShardingTransformType transformType) {
  Value input = dataFlowEdge.getInput();
  return castOwningShardableDataFlowOp(input).transformTargetSharding(
      input, sharding, transformType);
}

SmallVector<Value> getDataFlowSources(DataFlowEdgeOp dataFlowEdge) {
  Value input = dataFlowEdge.getInput();
  return castOwningShardableDataFlowOp(input).getEdgeSources(input);
}

SmallVector<Value> getNonEdgeOwnerTargets(DataFlowEdgeOp dataFlowEdge) {
  Value input = dataFlowEdge.getInput();
  return castOwningShardableDataFlowOp(input).getNonEdgeOwnerTargets(input);
}

}  // namespace sdy
}  // namespace mlir
