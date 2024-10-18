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

#include <cassert>
#include <memory>  // IWYU pragma: keep

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_APPLYSHARDINGCONSTRAINTSPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

// Returns true if `input` is used by any `ShardingConstraintOp` or
// `ManualComputationOp`, that has a different sharding than `sharding`.
bool isUsedByConstraintWithDifferentSharding(Value input,
                                             TensorShardingAttr sharding) {
  return llvm::any_of(input.getUses(), [&](OpOperand& use) {
    if (auto otherShardingConstraint =
            dyn_cast<ShardingConstraintOp>(use.getOwner())) {
      return otherShardingConstraint.getSharding() != sharding;
    }
    if (auto manualComputation =
            dyn_cast<ManualComputationOp>(use.getOwner())) {
      return manualComputation.getInSharding(use.getOperandNumber()) !=
             sharding;
    }
    return false;
  });
}

// Returns true if `input` is used by any `ShardingConstraintOp` or
// `ManualComputationOp`, that isn't `optionalShardingConstraint` if provided.
bool isUsedByOtherShardingConstraint(
    Value input, Operation* optionalShardingConstraint = nullptr) {
  return llvm::any_of(
      input.getUsers(), [optionalShardingConstraint](Operation* user) {
        return user != optionalShardingConstraint &&
               isa<ShardingConstraintOp, ManualComputationOp>(user);
      });
}

// Returns true if `input` should have its sharding set to `sharding` of the
// sharding constraint `op`.
bool shouldApply(Value input, TensorShardingAttr sharding, Operation* op) {
  if (getSharding(input) || input.getDefiningOp<DataFlowEdgeOp>()) {
    // `input` already has a sharding or is produced by a `DataFlowEdgeOp`.
    return false;
  }

  // This condition isn't fundamentally needed, but is here for parity with
  // GSPMD, we should revisit in the future.
  // TODO(b/358627707): revisit this condition.
  if (!sharding.isFullyClosed()) {
    // `sharding` has an open dimension.
    return false;
  }

  // TODO(b/358627707): revisit restricting to a single use if not dangling.
  // Return true if `input` has no other uses of type `ShardingConstraintOp` or
  // `ManualComputationOp` with a different sharding.
  return !isUsedByConstraintWithDifferentSharding(input, sharding);
}

// If `curShardingConstraintOp` is the last `ShardingConstraintOp` in a chain
// of `ShardingConstraintOp`s that holds the following constraints:
//
// 1. `curShardingConstraintOp` isn't used by any `ShardingConstraintOp` or
//    `ManualComputationOp` (otherwise it's not the last in the chain).
// 2. None of the other ops in the chain have more than one use.
//
// returns the first ShardingConstraintOp in that chain. Otherwise, returns
// nullptr.
//
// For example:
//
//   ```mlir
//   y = sdy.sharding_constraint(x)
//   z = sdy.sharding_constraint(y)
//   w = sdy.sharding_constraint(z)
//   ```
// Such that `w` isn't used by any other `ShardingConstraintOp` or
// `ManualComputationOp`, and `y` & `z` have a single use, in which case this
// method returns `y`.
ShardingConstraintOp getFirstShardingConstraintInChain(
    ShardingConstraintOp curShardingConstraintOp) {
  if (isUsedByOtherShardingConstraint(curShardingConstraintOp)) {
    return nullptr;
  }

  ShardingConstraintOp prevShardingConstraintOp = curShardingConstraintOp;
  while ((curShardingConstraintOp =
              curShardingConstraintOp.getInput()
                  .getDefiningOp<ShardingConstraintOp>())) {
    if (!curShardingConstraintOp->hasOneUse()) {
      return nullptr;
    }
    prevShardingConstraintOp = curShardingConstraintOp;
  }

  return prevShardingConstraintOp;
}

void moveAfterValue(Operation* op, Value value) {
  if (Operation* defOp = value.getDefiningOp()) {
    op->moveAfter(defOp);
  } else {
    Block* block = cast<BlockArgument>(value).getOwner();
    op->moveBefore(block, block->begin());
  }
}

// Moves a chain of `ShardingConstraintOp`s ending with `shardingConstraintOp`
// after `value`.
void moveChainAfterValue(ShardingConstraintOp shardingConstraintOp,
                         Value value) {
  while (shardingConstraintOp) {
    moveAfterValue(shardingConstraintOp, value);
    shardingConstraintOp =
        shardingConstraintOp.getInput().getDefiningOp<ShardingConstraintOp>();
  }
}

struct ApplyShardingConstraintsPass
    : public impl::ApplyShardingConstraintsPassBase<
          ApplyShardingConstraintsPass> {
  using ApplyShardingConstraintsPassBase::ApplyShardingConstraintsPassBase;

  void runOnOperation() final {
    getOperation().walk([](Operation* op) {
      TypeSwitch<Operation*>(op)
          .Case<ShardingConstraintOp>(
              [](ShardingConstraintOp shardingConstraintOp) {
                Value input = shardingConstraintOp.getInput();
                TensorShardingAttr sharding =
                    shardingConstraintOp.getSharding();
                if (shouldApply(input, sharding, shardingConstraintOp)) {
                  setSharding(input, sharding);
                }

                // If `shardingConstraintOp` is the last op in a chain of at
                // least two sharding constraints, and the input of the chain
                // isn't used by any other sharding constraint, then replace
                // all uses of the input with `shardingConstraintOp`.
                if (ShardingConstraintOp firstInChain =
                        getFirstShardingConstraintInChain(shardingConstraintOp);
                    firstInChain && firstInChain != shardingConstraintOp &&
                    !isUsedByOtherShardingConstraint(firstInChain.getInput(),
                                                     firstInChain)) {
                  // We need to move the chain right after its input, to make
                  // sure all other uses of the input are after the chain.
                  moveChainAfterValue(shardingConstraintOp,
                                      firstInChain.getInput());
                  firstInChain.getInput().replaceAllUsesExcept(
                      shardingConstraintOp.getResult(), firstInChain);
                }
              })
          .Case<ManualComputationOp>(
              [](ManualComputationOp manualComputationOp) {
                for (auto [operand, sharding] : llvm::zip_equal(
                         manualComputationOp.getOperands(),
                         manualComputationOp.getInShardings().getShardings())) {
                  if (shouldApply(operand, sharding, manualComputationOp)) {
                    setSharding(operand, sharding);
                  }
                }
              });
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
