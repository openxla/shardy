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
#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/debugging/source_sharding.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

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
// `ManualComputationOp`, that isn't `excluded` if provided.
bool isUsedByShardingConstraint(Value input, Operation* excluded = nullptr) {
  return llvm::any_of(input.getUsers(), [excluded](Operation* user) {
    return user != excluded &&
           isa<ShardingConstraintOp, ManualComputationOp>(user);
  });
}

void moveAfterValue(Operation* op, Value value) {
  if (Operation* defOp = value.getDefiningOp()) {
    op->moveAfter(defOp);
  } else {
    // Move to front of block.
    Block* block = cast<BlockArgument>(value).getOwner();
    op->moveBefore(block, block->begin());
  }
}

// Returns true if `input` should have its sharding set to `sharding` of a
// sharding constraint.
bool shouldApply(Value input, TensorShardingAttr sharding) {
  if (!getShardableValue(input)) {
    // A sharding can't be attached to `input`, it's likely a scalar block arg.
    return false;
  }
  if (getSharding(input)) {
    // `input` already has a sharding.
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

// Applies the `sharding` of a sharding constraint to `input` if `shouldApply`
// returns true.
//
// If `input` is a target of a data-flow edge (see `DataFlowEdgeOp::lookup`),
// then instead of setting the ops's sharding to `sharding`, we replace all uses
// of `input` with the `ShardingConstraintOp` returned by
// `getConstraintAfterValue`. This is to avoid restricting the sharding of all
// targets of the edge and to match GSPMD.
void applyConstraint(
    Operation* op, Value input, TensorShardingAttr sharding,
    std::function<ShardingConstraintOp()> getConstraintAfterValue) {
  if (!shouldApply(input, sharding)) {
    return;
  }

  if (input.getDefiningOp<DataFlowEdgeOp>() || DataFlowEdgeOp::lookup(input)) {
    ShardingConstraintOp shardingConstraintOp = getConstraintAfterValue();
    input.replaceAllUsesExcept(shardingConstraintOp, shardingConstraintOp);
    // If `sharding` has unreduced axes, we need to set then on the sharding of
    // `input` directly, as unreduced axes don't propagate.
    if (!sharding.getUnreducedAxes().empty()) {
      setSharding(input,
                  getOrCreateSharding(input, sharding.getMeshOrRef())
                      .replaceUnreducedAxes(sharding.getUnreducedAxes()));
    }
  } else {
    MLIRContext* context = op->getContext();
    OpShardingRuleAttr shardingRule =
        getOrCreateShardingRule(op, /*conservativePropagation=*/false,
                                /*set_sharding_rule_attr=*/true);

    if (context->hasActionHandler() && shardingRule) {
      ValueRange operands = op->getOperands();
      ValueRange results = op->getResults();
      SmallVector<TensorShardingAttr> operandShardings = getShardings(operands);
      SmallVector<TensorShardingAttr> resultShardings = getShardings(results);
      MeshAttr mesh = getCommonMesh(operandShardings, resultShardings, op);
      ShardingProjection shardingProjection = ShardingProjection::build(
          operandShardings, resultShardings, shardingRule, mesh);

      bool anyUpdated = false;
      auto updateShardings = [&]() {
        setSharding(input, sharding);
        ShardingProjection newShardingProjection = ShardingProjection::build(
            getShardings(operands), getShardings(results), shardingRule, mesh);
        anyUpdated = newShardingProjection != shardingProjection;
        shardingProjection = newShardingProjection;
      };

      context->executeAction<SourceShardingAction>(
          updateShardings,
          /*IRUnits=*/{op}, op, operands, results, mesh, shardingRule,
          shardingProjection,
          /*anyUpdated=*/anyUpdated);
    } else {
      setSharding(input, sharding);
    }
  }
}

// Given the head of a `ShardingConstraintOp` chain, returns the tail of the
// chain if all conditions are true. Otherwise, returns nullptr.
//
// 1. The `head` should be the first one in the chain.
// 2. The input of the `head` cannot have a sharding.
// 3. The input of the `head` isn't used by any other `ShardingConstraintOp` or
//    `ManualComputationOp`.
// 4. Other than `tail`, none of the ops in the chain have more than one use.
// 5. The length of the chain is at least 2.
//
// For example:
//
//   ```mlir
//   y = sdy.sharding_constraint(x)
//   z = sdy.sharding_constraint(y)
//   w = sdy.sharding_constraint(z)
//   ```
// `y` is the head of the chain, and `w` is the tail. Taking `y` as the input,
// we return `w` if the following conditions are true.
// 1. `y` and `z` does not have any other uses.
// 2. `x` does not have a sharding.
// 3. `x` isn't used by any other `ShardingConstraintOp` or
//    `ManualComputationOp`.
//
// TODO(b/377454801): reconsider this logic.
ShardingConstraintOp getTailOfShardingConstraintChain(
    ShardingConstraintOp head) {
  if (head.getInput().getDefiningOp<ShardingConstraintOp>()) {
    // `head` is not the first one in the chain.
    return nullptr;
  }

  if (getSharding(head.getInput())) {
    // The input of the `head` already has a sharding.
    return nullptr;
  }

  if (isUsedByShardingConstraint(head.getInput(), /*excluded=*/head)) {
    return nullptr;
  }

  ShardingConstraintOp tail = head;
  while (tail->hasOneUse() && isa<ShardingConstraintOp>(*tail->user_begin())) {
    tail = cast<ShardingConstraintOp>(*tail->user_begin());
  }

  if (tail == head) {
    // The chain is of length 1.
    return nullptr;
  }

  if (isUsedByShardingConstraint(tail)) {
    // `tail` is not the last one in the chain. We achieve this since `tail` has
    // multiple uses.
    return nullptr;
  }

  return tail;
}

struct ApplyShardingConstraintsPass
    : public impl::ApplyShardingConstraintsPassBase<
          ApplyShardingConstraintsPass> {
  using ApplyShardingConstraintsPassBase::ApplyShardingConstraintsPassBase;

  ApplyShardingConstraintsPass() = default;

  // Copy constructor to handle Option members
  ApplyShardingConstraintsPass(const ApplyShardingConstraintsPass& other)
      : ApplyShardingConstraintsPassBase<ApplyShardingConstraintsPass>(other) {
    debugShardingOrigins = other.debugShardingOrigins;
    debugPropagationEdges = other.debugPropagationEdges;
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext& context = getContext();

    // Prepare debugging handler for sharding origins and edge sources.
    ShardingDebugMappings mappings(debugShardingOrigins,
                                   debugPropagationEdges);
    SourceShardingHandler handler(&mappings);
    // Prepare the handler and register it to the context.
    handler.prepareHandler(moduleOp);

    OpBuilder builder(&context);
    moduleOp.walk([&](Operation* op) {
      TypeSwitch<Operation*>(op)
          .Case<ShardingConstraintOp>(
              [](ShardingConstraintOp shardingConstraintOp) {
                // If `getTailOfShardingConstraintChain` returns a non-null
                // value, we replace all uses of `head`s input that:
                // 1. Aren't a `func.return` op.
                // 2. Are defined after `tail` (and in same block) with `tail`.
                //
                // Refer to `getTailOfShardingConstraintChain` for more details.
                ShardingConstraintOp head = shardingConstraintOp;
                if (ShardingConstraintOp tail =
                        getTailOfShardingConstraintChain(head)) {
                  head.getInput().replaceUsesWithIf(
                      tail.getResult(), [&](OpOperand& use) {
                        return use.getOwner() != head &&
                               !isa<func::ReturnOp>(use.getOwner()) &&
                               tail->getBlock() == use.getOwner()->getBlock() &&
                               tail->isBeforeInBlock(use.getOwner());
                      });
                }

                Value input = shardingConstraintOp.getInput();
                TensorShardingAttr sharding =
                    shardingConstraintOp.getSharding();
                applyConstraint(shardingConstraintOp, input, sharding,
                                /*getConstraintAfterValue=*/[&]() {
                                  moveAfterValue(shardingConstraintOp, input);
                                  return shardingConstraintOp;
                                });
              })
          .Case<ManualComputationOp>(
              [&](ManualComputationOp manualComputationOp) {
                for (auto [operand, sharding] : llvm::zip_equal(
                         manualComputationOp.getOperands(),
                         manualComputationOp.getInShardings().getShardings())) {
                  applyConstraint(
                      manualComputationOp, operand, sharding,
                      /*getConstraintAfterValue=*/
                      [&, operand = operand, sharding = sharding]() {
                        // We can't move the `ManualComputationOp`, so we create
                        // a new `ShardingConstraintOp` after the operand.
                        builder.setInsertionPointAfterValue(operand);
                        return builder.create<ShardingConstraintOp>(
                            manualComputationOp.getLoc(), operand, sharding);
                      });
                }
              });
    });

    // Unregister the handler and save the sharding origins on the module.
    context.registerActionHandler(nullptr);
    handler.saveOnModule(moduleOp);
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
