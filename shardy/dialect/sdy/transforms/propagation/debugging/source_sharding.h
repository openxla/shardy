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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_DEBUGGING_SOURCE_SHARDING_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_DEBUGGING_SOURCE_SHARDING_H_

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Unit.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// The type of source of a user sharding.
enum OriginShardingType {
  INPUT,       // User sharding came from a `FuncOp` `BlockArgument`.
  CONSTRAINT,  // User sharding came from an `sdy.sharding_constraint`.
  OUTPUT,      // User sharding came from a `FuncOp` `ResultAttr`.
  MC_INPUT,    // User sharding came from an `sdy.ManualComputationOp` input.
  MC_OUTPUT    // User sharding came from an `sdy.ManualComputationOp` output.
};

// The source of a user sharding.
struct OriginSharding {
  OriginShardingType type;
  int64_t index;
  // If the origin is a `sdy.sharding_constraint` or `sdy.ManualComputationOp`,
  // this is the unique id used in the `kShardingOriginNameAttr` to identify
  // the origin.
  int64_t sourceId = 0;
};

// Specifies whether a sharding came from an operand or a result.
enum EdgeNodeType { OPERAND, RESULT };

// The operand/result a sharding came from through an `Operation` to modify the
// sharding of some `Value` in the `Operation`.
struct EdgeNode {
  EdgeNodeType type;
  int64_t index;
};

// The source and target of a source sharding edge.
struct PropagationEdge {
  EdgeNode source;
  EdgeNode target;
  int64_t propagationStep;
};

// Types used for `debugPropagationEdgeSharding`.
using AxisToEdgeMap = llvm::DenseMap<AxisRefAttr, PropagationEdge>;
using AxisToEdgesMap =
    llvm::DenseMap<AxisRefAttr, SmallVector<PropagationEdge>>;
using OperationToEdgesMap = llvm::DenseMap<Operation*, AxisToEdgesMap>;
// Mapping from `FuncOp` to the edges for each result.
using FuncResultToEdgesMap =
    llvm::DenseMap<func::FuncOp, SmallVector<AxisToEdgesMap>>;

// Types used for `debugShardingOrigins`.
using AxisToOriginShardingMap = llvm::DenseMap<AxisRefAttr, OriginSharding>;
using ValueToOriginShardingMap = llvm::DenseMap<Value, AxisToOriginShardingMap>;

// The mappings used for debugging sharding origins and edge sources.
struct ShardingDebugMappings {
  ShardingDebugMappings(bool debugShardingOrigins,
                        bool debugPropagationEdgeSharding);

  // We do not allow copying of the mappings, as we don't want the mappings
  // to be copied over to the new instance. There should only ever be one
  // instance of the mappings object passed around as a pointer.
  ShardingDebugMappings(const ShardingDebugMappings&) = delete;
  ShardingDebugMappings& operator=(const ShardingDebugMappings&) = delete;

  bool debugShardingOrigins, debugPropagationEdgeSharding;
  OperationToEdgesMap operationToEdgesMap;
  // NOTE: we need a separate map for `FuncOp` results as propagation is run
  // per terminator operand/result pair, so we need to figure out which index
  // the `FuncOp` result is. So this saves the edges for each `FuncOp` result
  // separately.
  FuncResultToEdgesMap funcResultToEdgesMap;
  ValueToOriginShardingMap valueToOriginShardingMap;
};

// An MLIR action that saves what was the source of an axis to appear on the
// sharding of a given `Value`. Source here means a user defined sharding either
// on `FuncOp` inputs/outputs, an `sdy.sharding_constraint`, or
// `sdy.ManualComputationOp` input/output.
class SourceShardingAction : public tracing::ActionImpl<SourceShardingAction> {
 public:
  using Base = tracing::ActionImpl<SourceShardingAction>;

  SourceShardingAction(ArrayRef<IRUnit> irUnits, Operation* op,
                       ValueRange operands, ValueRange results, MeshAttr mesh,
                       OpShardingRuleAttr shardingRule,
                       const ShardingProjection& shardingProjection,
                       const bool& anyUpdated)
      : Base(irUnits),
        op(op),
        operands(operands),
        results(results),
        mesh(mesh),
        shardingRule(shardingRule),
        oldShardingProjection(shardingProjection),
        newShardingProjection(shardingProjection),
        anyUpdated(anyUpdated) {}

  static constexpr StringLiteral tag = "SourceShardingAction";
  static constexpr StringLiteral desc =
      "Stores the source of an axis to appear on a `Value`. Source here means "
      "a user defined sharding either on `FuncOp` inputs/outputs, an "
      "`sdy.sharding_constraint`, or `sdy.ManualComputationOp` input/output.";

  Operation* op;
  ValueRange operands, results;
  MeshAttr mesh;
  OpShardingRuleAttr shardingRule;
  // NOTE: `oldShardingProjection` is a copy while `newShardingProjection` is a
  // reference as when the action is executed, we want to see how the old and
  // new sharding projections differ.
  const ShardingProjection oldShardingProjection;
  const ShardingProjection& newShardingProjection;
  // Whether any of the operands/results were updated.
  const bool& anyUpdated;
};

// Handles `SourceShardingAction`s, figuring out what operand/result shardings
// have been propagated through due to new axes. Saves what was the source of
// the axis to appear on the sharding of a given `Value` to
// `operationToEdgesMap`/`funcResultToEdgesMap` and `valueToOriginShardingMap`.
struct SourceShardingHandler {
  SourceShardingHandler(ShardingDebugMappings* mappings);

  // If `action` is a `SourceShardingAction`, saves the source sharding
  // information on the module after `transform` has been applied.
  void operator()(function_ref<void()> transform,
                  const tracing::Action& action);

  // prepares the `SourceShardingHandler` for the `moduleOp` and registers it
  // as an action handler on the `MLIRContext` of `moduleOp`.
  void prepareHandler(ModuleOp moduleOp);

  // Saves the sharding origin and edge source information on the `moduleOp`.
  void saveOnModule(ModuleOp moduleOp);

 private:
  ShardingDebugMappings* mappings;
  int64_t propagationStep = 0;
};

// Saves an array of all the origin sharding and propagation edge dictionaries
// for the given `edgeOwners` on `op`. If non exist, nothing is saved.
//
// Saving the info depends on if the corresponding `sinkDebugShardingOrigins`
// and `sinkDebugPropagationEdgeSharding` are true.
//
// For debugging the origin shardings and propagation edges, we want to preserve
// the debugging dictionaries from the `DataFlowEdgeOp`s on the owning op so
// that they are preserved after the propagation pipeline.
//
// See the `debug-sharding-origins` and `debug-edge-source-sharding` config on
// propagation for more details.
void saveDebugInfoDictsFromDataFlowEdges(ValueRange edgeOwners, Operation* op,
                                         bool sinkDebugShardingOrigins,
                                         bool sinkDebugPropagationEdgeSharding,
                                         EdgeNodeType edgeNodeType,
                                         IRRewriter& rewriter);

}  // namespace sdy
}  // namespace mlir

#endif  // SRC_SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_DEBUGGING_SOURCE_SHARDING_H_
