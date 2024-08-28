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

#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/ir/data_flow_utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_propagation_context.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_BASICPROPAGATIONPASS
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

namespace {

// Sets the sharding of a tensor at a given index to the given
// `TensorShardingAttr`.
using SetShardingPerTensorCallback =
    std::function<void(TensorShardingAttr, int64_t)>;

// Sets the sharding of a tensor to the given `TensorShardingAttr`.
using SetTensorShardingCallback = std::function<void(TensorShardingAttr)>;

using NotifyOpModifiedCallback = std::function<void(Operation*)>;

// Propagates the sharding of an operation (between operands and results) that
// has a registered or custom `OpShardingRuleAttr`.
class PropagateRegisteredOp : public RewritePattern {
 public:
  explicit PropagateRegisteredOp(
      MLIRContext* context, GetDirectionToPropagateFn getDirectionToPropagate,
      bool conservativePropagation, const FactorPropagation& factorPropagation,
      const ShardingPropagationContext& shardingPropagationContext)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        getDirectionToPropagate(getDirectionToPropagate),
        conservativePropagation(conservativePropagation),
        factorPropagation(factorPropagation),
        shardingPropagationContext(shardingPropagationContext) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    OpShardingRuleAttr shardingRule =
        getOrCreateShardingRule(op, conservativePropagation);
    if (!shardingRule) {
      // Rule doesn't exist for ops that aren't known/registered.
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "op doesn't have a registered sharding rule";
      });
    }
    PropagationDirection direction = getDirectionToPropagate(op);
    if (direction == PropagationDirection::NONE) {
      // No need to continue to propagate if the direction is `NONE`, as
      // neither operands nor results can be updated.
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "propagation direction on op is NONE";
      });
    }

    return shardingPropagationContext
        .propagateTensorShardingsWithDefaultCallbacks(
            op->getOperands(), op->getResults(), shardingRule, op, rewriter,
            factorPropagation, direction, conservativePropagation);
  }

 private:
  GetDirectionToPropagateFn getDirectionToPropagate;
  bool conservativePropagation;
  const FactorPropagation& factorPropagation;
  const ShardingPropagationContext& shardingPropagationContext;
};

// Propagates shardings between the sources and targets of an
// `sdy.data_flow_edge`.
//
// The `sdy.data_flow_edge` holds the updateable sharding of all targets.
class PropagateDataFlowEdgeOp : public OpRewritePattern<DataFlowEdgeOp> {
 public:
  explicit PropagateDataFlowEdgeOp(
      MLIRContext* context, const FactorPropagation& factorPropagation,
      const ShardingPropagationContext& shardingPropagationContext)
      : OpRewritePattern<DataFlowEdgeOp>(context),
        factorPropagation(factorPropagation),
        shardingPropagationContext(shardingPropagationContext) {}

  LogicalResult matchAndRewrite(DataFlowEdgeOp dataFlowEdgeOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<Value> sources = getDataFlowSources(dataFlowEdgeOp);
    // The sharding of `dataFlowEdgeOp.getResult()` is the sharding of all
    // targets.
    return shardingPropagationContext
        .propagateTensorShardingsWithDefaultCallbacks(
            sources, dataFlowEdgeOp.getResult(),
            createIdentityShardingRule(
                cast<ShapedType>(dataFlowEdgeOp.getType()), sources.size()),
            dataFlowEdgeOp, rewriter, factorPropagation);
  }

 private:
  const FactorPropagation& factorPropagation;
  const ShardingPropagationContext& shardingPropagationContext;
};

// The `ManualComputationOp` has no sharding rule, and also has a body, so like
// `WhileOp` and `CaseOp`, we need a special rewrite pattern for it. Given the
// following:
// ```
// y_0, ..., y_n = sdy.manual_computation (x_0, ..., x_n)
//                 in_shardings=[...],
//                 out_shardings=[...],
//                 manual_axes={...},
//                 ((body_arg_0,..., body_arg_n) {
//                   %inside = op(body_arg_0)
//                   ...
//                   sdy.return result_arg_0, ..., result_arg_n
//                 })
// ...
// %z = op(..., y_i, ...)
// ```
// This pattern propagates between:
// - `x_i`/`in_shardings[i]` and `body_arg_i`,
// - `result_arg_i` and `y_i`/`out_shardings[i]`
//
// This makes sure any sharding inside the body propagates out. For propagating
// an `in_sharding` into the body, e.g. to `%inside`, then the
// `PropagateRegisteredOp` RewritePattern handles this case. Same for
// propagating from `y_i` to `%z`.
class PropagateManualComputationOp
    : public OpRewritePattern<ManualComputationOp> {
 public:
  explicit PropagateManualComputationOp(
      MLIRContext* context, const FactorPropagation& factorPropagation,
      const ShardingPropagationContext& shardingPropagationContext)
      : OpRewritePattern<ManualComputationOp>(context),
        factorPropagation(factorPropagation),
        shardingPropagationContext(shardingPropagationContext) {}

  LogicalResult matchAndRewrite(ManualComputationOp manualComputationOp,
                                PatternRewriter& rewriter) const override {
    bool updated = false;

    // 1. Propagate between the operands of the `ManualComputationOp` and the
    //    block arguments (specifically the `in_shardings`, but we use the op's
    //    block arguments as an alias for them).
    for (BlockArgument blockArg :
         manualComputationOp.getBody().getArguments()) {
      const int64_t argNumber = blockArg.getArgNumber();
      Value operand = manualComputationOp->getOperand(argNumber);
      updated |=
          shardingPropagationContext
              .propagateSingleTensorSharding(
                  operand, blockArg,
                  // Since this is propagating outside of the region of the
                  // `ManualComputationOp`, make sure we keep the manual axes
                  // as we may be able to propagate those backwards.
                  // `getSharding` on the block arg would remove them, so
                  // need to get the right `in_shardings` explicitly using
                  // `getInSharding`.
                  getSharding(operand),
                  manualComputationOp.getInSharding(argNumber),
                  [&operand](TensorShardingAttr sharding) {
                    setSharding(operand, sharding);
                  },
                  // Similarly as above, since `setSharding` will add the
                  // manual axes back, but they already exist, we set the
                  // `in_shardings` explicitly using `setInSharding`.
                  [&manualComputationOp,
                   argNumber](TensorShardingAttr sharding) {
                    manualComputationOp.setInSharding(argNumber, sharding);
                  },
                  createIdentityShardingRule(
                      cast<RankedTensorType>(operand.getType())),
                  manualComputationOp, &rewriter, factorPropagation)
              .succeeded();
    }

    // 2. Propagate between the uses of the `ManualComputationOp` and the
    //    terminator of the body.
    for (OpOperand& returnValue :
         getBodyTerminatorOpOperands(manualComputationOp)) {
      const int64_t operandNumber = returnValue.getOperandNumber();
      OpResult opResult = manualComputationOp->getResult(operandNumber);
      // Since this is propagating on the border of the local region of manual
      // axes and global program, only use shardings without the manual axes.
      // `setSharding` will add them back for `out_shardings`.
      updated |=
          shardingPropagationContext
              .propagateSingleTensorSharding(
                  returnValue.get(), opResult, getSharding(returnValue.get()),
                  manualComputationOp.getOutShardingWithoutManualAxes(
                      operandNumber),
                  [&returnValue](TensorShardingAttr sharding) {
                    setSharding(returnValue.get(), sharding);
                  },
                  [&](TensorShardingAttr sharding) {
                    manualComputationOp.setOutShardingAddingManualAxes(
                        operandNumber, sharding);
                  },
                  createIdentityShardingRule(
                      cast<RankedTensorType>(opResult.getType())),
                  manualComputationOp, &rewriter, factorPropagation)
              .succeeded();
    }

    return success(updated);
  }

 private:
  const FactorPropagation& factorPropagation;
  const ShardingPropagationContext& shardingPropagationContext;
};

// Propagates through a `PropagationBarrierOp` accounting for the direction in
// which it blocks propagation.
class PropagatePropagationBarrier
    : public OpRewritePattern<PropagationBarrierOp> {
 public:
  explicit PropagatePropagationBarrier(
      MLIRContext* context, const FactorPropagation& factorPropagation,
      const ShardingPropagationContext& shardingPropagationContext)
      : OpRewritePattern<PropagationBarrierOp>(context),
        factorPropagation(factorPropagation),
        shardingPropagationContext(shardingPropagationContext) {}

  LogicalResult matchAndRewrite(PropagationBarrierOp propagationBarrierOp,
                                PatternRewriter& rewriter) const override {
    return shardingPropagationContext
        .propagateTensorShardingsWithDefaultCallbacks(
            propagationBarrierOp.getInput(), propagationBarrierOp.getResult(),
            createIdentityShardingRule(
                cast<RankedTensorType>(propagationBarrierOp.getType())),
            propagationBarrierOp, rewriter, factorPropagation,
            propagationBarrierOp.getAllowedDirection());
  }

 private:
  const FactorPropagation& factorPropagation;
  const ShardingPropagationContext& shardingPropagationContext;
};

// The basic propagation pass that uses the default implementation of
// `BasicPropagationPassImpl`.
struct BasicPropagationPass
    : public impl::BasicPropagationPassBase<BasicPropagationPass> {
  using BasicPropagationPassBase::BasicPropagationPassBase;

  // NOLINTBEGIN(clang-diagnostic-shadow-field)
  explicit BasicPropagationPass(bool keepShardingRules, StringRef dumpDirectory,
                                bool conservativePropagation) {
    // NOLINTEND(clang-diagnostic-shadow-field)
    this->keepShardingRules = keepShardingRules;
    this->dumpDirectory = dumpDirectory.str();
    this->conservativePropagation = conservativePropagation;
  }
};

}  // namespace

PropagationDirection propagateAny(Operation*) {
  return PropagationDirection::BOTH;
}

LogicalResult BasicPropagationPassImpl::propagate(
    ModuleOp moduleOp, const FactorPropagation& factorPropagation,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  // Pushes any shardings that exist on the `funcOp` result type attrs to the
  // corresponding values returned in the terminator of the body of `funcOp`.
  if (failed(shardingPropagationContext.propagateAllFuncResultsInModule(
          moduleOp, factorPropagation))) {
    return failure();
  }

  MLIRContext* context = moduleOp.getContext();
  RewritePatternSet patterns(context);
  patterns.add<PropagateDataFlowEdgeOp, PropagateManualComputationOp,
               PropagatePropagationBarrier>(context, factorPropagation,
                                            shardingPropagationContext);
  patterns.add<PropagateRegisteredOp>(
      context, getDirectionToPropagate, conservativePropagation,
      factorPropagation, shardingPropagationContext);
  // Note that we only need a single iteration (and another to confirm
  // convergence), since we make sure ops whose sharding changes are
  // added back to the worklist.
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns),
                                          config))) {
    return failure();
  }

  // Pushes any shardings from the values returned in the terminator of the body
  // of `funcOp` to the corresponding `funcOp` result type attrs.
  if (failed(shardingPropagationContext.propagateAllFuncResultsInModule(
          moduleOp, factorPropagation))) {
    return failure();
  }
  return success();
}

LogicalResult BasicPropagationPassImpl::propagate(
    ModuleOp moduleOp, GetDirectionToPropagateFn getDirectionToPropagate) {
  return propagate(moduleOp, basicFactorPropagation, getDirectionToPropagate);
}

void BasicPropagationPassImpl::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (failed(propagate(moduleOp))) {
    signalPassFailure();
    return;
  }
  if (!keepShardingRules) {
    removeShardingRules(moduleOp);
  }
  saveModuleOp(moduleOp, dumpDirectory, "sdy_module_after_propagation");
}

std::unique_ptr<Pass> createBasicPropagationPass(bool keepShardingRules,
                                                 StringRef dumpDirectory,
                                                 bool conservativePropagation) {
  return std::make_unique<BasicPropagationPass>(
      keepShardingRules, dumpDirectory, conservativePropagation);
}

}  // namespace sdy
}  // namespace mlir
