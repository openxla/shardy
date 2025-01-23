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

#ifndef SHARDY_DIALECT_SDY_IR_DATA_FLOW_UTILS_H_
#define SHARDY_DIALECT_SDY_IR_DATA_FLOW_UTILS_H_

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// TODO(tomnatan): methods can be moved to DataFlowEdgeOp.

// See `ShardableDataFlowOpInterface` documentation for more information on
// data-flow ops and edges.

// If `target` is a target of a data-flow edge, returns the corresponding
// `DataFlowEdgeOp`, otherwise returns `nullptr`.
DataFlowEdgeOp getDataFlowEdge(Value target);

// If `source` is a source of a data-flow edge, returns the corresponding
// `DataFlowEdgeOp`, otherwise returns `nullptr`.
DataFlowEdgeOp getDataFlowEdge(OpOperand& source);

// Transforms the `sharding` depending on `transformType`.
//
// See `DataFlowShardingTransformType` for more information.
TensorShardingAttr transformTargetSharding(
    DataFlowEdgeOp dataFlowEdge, TensorShardingAttr sharding,
    DataFlowShardingTransformType transformType);

// Returns all sources of the given `dataFlowEdge`.
SmallVector<Value> getDataFlowSources(DataFlowEdgeOp dataFlowEdge);

// Returns all non-edge-owner targets of the given `dataFlowEdge`.
SmallVector<Value> getNonEdgeOwnerTargets(DataFlowEdgeOp dataFlowEdge);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_DATA_FLOW_UTILS_H_
