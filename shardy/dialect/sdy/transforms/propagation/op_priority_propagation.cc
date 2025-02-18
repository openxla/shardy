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

#include "shardy/dialect/sdy/transforms/propagation/op_priority_propagation.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/op_properties.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_OPPRIORITYPROPAGATIONPASS
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

namespace {

// A function that determines in which direction propagation should happen for a
// given op and factor index.
using GetDirectionToPropagateFnPtr = PropagationDirection (*)(Operation*,
                                                              int64_t);

PropagationDirection isPassThrough(Operation* op, int64_t) {
  if (isElementwise(op) ||
      isa<stablehlo::ReshapeOp, stablehlo::TransposeOp, DataFlowEdgeOp>(op)) {
    return PropagationDirection::BOTH;
  }
  if (isa<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp>(op)) {
    return PropagationDirection::FORWARD;
  }
  return PropagationDirection::NONE;
}

// NOTE: if the `op` has no sharding rule, then we will assume it uses an
// identity sharding rule. For example, `DataFlowEdgeOp`.
PropagationDirection onlyPassThroughFactors(Operation* op,
                                            int64_t factorIndex) {
  if (auto shardingRule =
          op->getAttrOfType<OpShardingRuleAttr>(kShardingRuleAttr);
      shardingRule && !shardingRule.isPassThroughFactor(factorIndex)) {
    return PropagationDirection::NONE;
  }
  return PropagationDirection::BOTH;
}

constexpr std::array<GetDirectionToPropagateFnPtr, 3> opPropagationSchedule = {
    isPassThrough, onlyPassThroughFactors, propagateAny};

// Returns the direction in which the given operation should be propagated.
//
// This will also take into account any `getDirectionToPropagate` passed through
// a caller. It will return the intersection of the passed in
// `getDirectionToPropagate` and the op based direction.
GetDirectionToPropagateFn getOpBasedDirectionToPropagate(
    int64_t currentPriority,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  return [currentPriority, getDirectionToPropagate](Operation* op,
                                                    int64_t factorIndex) {
    PropagationDirection opBasedDirection = std::accumulate(
        opPropagationSchedule.begin(),
        opPropagationSchedule.begin() + currentPriority + 1,
        PropagationDirection::NONE,
        [&](PropagationDirection acc, GetDirectionToPropagateFnPtr dirFn) {
          return unionOfPropagationDirections(acc, dirFn(op, factorIndex));
        });
    return intersectionOfPropagationDirections(
        opBasedDirection, getDirectionToPropagate(op, factorIndex));
  };
}

// The op-priority propagation pass that uses the default implementation of
// `OpPriorityPropagationPassImpl`.
struct OpPriorityPropagationPass
    : public impl::OpPriorityPropagationPassBase<OpPriorityPropagationPass> {
  using OpPriorityPropagationPassBase::OpPriorityPropagationPassBase;

  explicit OpPriorityPropagationPass(const PropagationOptions& options) {
    setPropagationOptions(options);
  }
};

}  // namespace

LogicalResult OpPriorityPropagationPassImpl::propagate(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    const ShardingGroupMap& shardingGroupMap,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  if (!runOpPriorityPropagation) {
    return AggressivePropagationPassImpl::propagate(
        moduleOp, symbolTable, shardingGroupMap, getDirectionToPropagate);
  }
  // Reset currentPriority to 0. Before running the pass. This same instance
  // could have been run earlier already (e.g. with a different user priority).
  for (int64_t currentPriority = 0;
       currentPriority < opPropagationSchedule.size(); currentPriority++) {
    if (AggressivePropagationPassImpl::propagate(
            moduleOp, symbolTable, shardingGroupMap,
            getOpBasedDirectionToPropagate(currentPriority,
                                           getDirectionToPropagate))
            .failed()) {
      return failure();
    }
  }
  return success();
}

std::unique_ptr<Pass> createOpPriorityPropagationPass(
    const PropagationOptions& options) {
  return std::make_unique<OpPriorityPropagationPass>(options);
}

}  // namespace sdy
}  // namespace mlir
