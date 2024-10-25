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
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/op_properties.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_OPPRIORITYPROPAGATIONPASS
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

namespace {

// A function that determines in which direction propagation should happen for a
// given op.
using GetDirectionToPropagateFnPtr = PropagationDirection (*)(Operation*);

template <typename... OpTs>
PropagationDirection isaBoth(Operation* op) {
  return isa<OpTs...>(op) ? PropagationDirection::BOTH
                          : PropagationDirection::NONE;
}

template <typename... OpTs>
PropagationDirection isNotABoth(Operation* op) {
  return !isa<OpTs...>(op) ? PropagationDirection::BOTH
                           : PropagationDirection::NONE;
}

template <typename... OpTs>
PropagationDirection isaForward(Operation* op) {
  return isa<OpTs...>(op) ? PropagationDirection::FORWARD
                          : PropagationDirection::NONE;
}

template <typename... OpTs>
PropagationDirection isaBackward(Operation* op) {
  return isa<OpTs...>(op) ? PropagationDirection::BACKWARD
                          : PropagationDirection::NONE;
}

PropagationDirection isPassThrough(Operation* op) {
  if (isElementwise(op) ||
      isa<stablehlo::ReshapeOp, stablehlo::TransposeOp>(op)) {
    return PropagationDirection::BOTH;
  }
  if (isa<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp>(op)) {
    return PropagationDirection::FORWARD;
  }
  return PropagationDirection::NONE;
}

constexpr std::array<GetDirectionToPropagateFnPtr, 4> opPropagationSchedule = {
    isPassThrough, isNotABoth<stablehlo::BroadcastInDimOp>,
    isaBackward<stablehlo::BroadcastInDimOp>, propagateAny};

// Returns the direction in which the given operation should be propagated.
//
// This will also take into account any `getDirectionToPropagate` passed through
// a caller. It will return the intersection of the passed in
// `getDirectionToPropagate` and the op based direction.
GetDirectionToPropagateFn getOpBasedDirectionToPropagate(
    int64_t currentOpPriority,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  return [currentOpPriority, getDirectionToPropagate](Operation* op) {
    PropagationDirection opBasedDirection = std::accumulate(
        opPropagationSchedule.begin(),
        opPropagationSchedule.begin() + currentOpPriority + 1,
        PropagationDirection::NONE,
        [&](PropagationDirection acc, GetDirectionToPropagateFnPtr dirFn) {
          return unionOfPropagationDirections(acc, dirFn(op));
        });
    return intersectionOfPropagationDirections(opBasedDirection,
                                               getDirectionToPropagate(op));
  };
}

// The op-priority propagation pass that uses the default implementation of
// `OpPriorityPropagationPassImpl`.
struct OpPriorityPropagationPass
    : public impl::OpPriorityPropagationPassBase<OpPriorityPropagationPass> {
  using OpPriorityPropagationPassBase::OpPriorityPropagationPassBase;

  explicit OpPriorityPropagationPass(
      // NOLINTBEGIN(clang-diagnostic-shadow-field)
      bool keepShardingRules, StringRef dumpDirectory,
      bool conservativePropagation, bool runOpPriorityPropagation) {
    // NOLINTEND(clang-diagnostic-shadow-field)
    this->keepShardingRules = keepShardingRules;
    this->dumpDirectory = dumpDirectory.str();
    this->conservativePropagation = conservativePropagation;
    this->runOpPriorityPropagation = runOpPriorityPropagation;
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
  // Reset currentOpPriority to 0. Before running the pass. This same instance
  // could have been run earlier already (e.g. with a different user priority).
  for (int64_t currentOpPriority = 0;
       currentOpPriority < opPropagationSchedule.size(); currentOpPriority++) {
    if (AggressivePropagationPassImpl::propagate(
            moduleOp, symbolTable, shardingGroupMap,
            getOpBasedDirectionToPropagate(currentOpPriority,
                                           getDirectionToPropagate))
            .failed()) {
      return failure();
    }
  }
  return success();
}

std::unique_ptr<Pass> createOpPriorityPropagationPass(
    bool keepShardingRules, StringRef dumpDirectory,
    bool conservativePropagation) {
  return std::make_unique<OpPriorityPropagationPass>(
      keepShardingRules, dumpDirectory, conservativePropagation,
      /*runOpPriorityPropagation=*/true);
}

}  // namespace sdy
}  // namespace mlir
