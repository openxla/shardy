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
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/ir/data_flow_utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/debugging/source_sharding.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_BASICPROPAGATIONPASS
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

namespace {

using func::FuncOp;

// Sets the sharding of a tensor at a given index to the given
// `TensorShardingAttr`.
using SetShardingPerTensorCallback =
    std::function<void(TensorShardingAttr, int64_t)>;

// Sets the sharding of a tensor to the given `TensorShardingAttr`.
using SetTensorShardingCallback = std::function<void(TensorShardingAttr)>;

using NotifyOpModifiedCallback = std::function<void(Operation*)>;

// Calls `notifyOpModified` on all users of `value`, so they will be added back
// to the worklist.
//
// Special cases:
// - If a use is a source of an `sdy.data_flow_edge` (e.g. while operand), add
//   the latter back to the worklist.
// - If a user is a terminator, the parent op will be added back to the worklist
//   instead of the terminator.
void notifyUsersModified(Value value,
                         NotifyOpModifiedCallback notifyOpModified) {
  for (OpOperand& use : value.getUses()) {
    Operation* user = use.getOwner();

    if (DataFlowEdgeOp dataFlowEdge = getDataFlowEdge(use)) {
      notifyOpModified(dataFlowEdge);
    } else if (user->hasTrait<OpTrait::IsTerminator>()) {
      notifyOpModified(user->getParentOp());
    } else {
      notifyOpModified(user);
    }
  }
}

// Calls `notifyOpModified` on all ops that are affected by changing the
// sharding of `value`, so that they will be added back to the worklist.
void notifyShardingModified(Value value,
                            NotifyOpModifiedCallback notifyOpModified) {
  if (auto dataFlowEdge = value.getDefiningOp<DataFlowEdgeOp>()) {
    for (Value nonEdgeOwnerTarget : getNonEdgeOwnerTargets(dataFlowEdge)) {
      notifyUsersModified(nonEdgeOwnerTarget, notifyOpModified);
    }
  }

  if (auto opResult = dyn_cast<OpResult>(value)) {
    // If the value has a defining op, add it back to the worklist.
    notifyOpModified(opResult.getOwner());
  } else {
    // Otherwise, the value is a block argument with an attached sharding, so
    // we need to add its parent op (e.g. manual computation) back to the
    // worklist.
    notifyOpModified(value.getParentBlock()->getParentOp());
  }

  // Notify that all users of `value` are being modified, so they will be
  // added back to the worklist as well.
  notifyUsersModified(value, notifyOpModified);
}

// Update the sharding of `value` to the sharding in `tensorFactorShardings`.
//
// Returns true if it's possible to update the sharding, i.e., if strided view
// isn't needed and all non-minor-most factors are divisible by sharding axes.
bool updateTensorSharding(
    TensorShardingAttr oldTensorSharding,
    SetTensorShardingCallback setTensorShardingCallback,
    const TensorFactorShardings& tensorFactorShardings,
    TensorMappingAttr tensorMapping, ArrayRef<int64_t> factorSizes,
    StringRef meshName, MeshAttr mesh, Value modifiedValue,
    const ShardingGroupMap& shardingGroupMap,
    std::optional<NotifyOpModifiedCallback> notifyOpModified) {
  // We can assume `modifiedValue` exists since we are updating its sharding.
  assert(modifiedValue && "modified value should exist");
  TensorShardingAttr newSharding =
      tensorFactorShardings.createTensorShardingAttr(
          mesh.getContext(), tensorMapping, factorSizes, meshName, mesh);
  // `oldTensorSharding` may be null if there is no sharding, in which case we
  // check if `newSharding` is empty.
  // TODO(tomnatan): remove this checking if the new sharding equals the old
  // sharding once strides is supported and divisibility is checked when finding
  // the compatible axes.
  if ((!oldTensorSharding && newSharding.emptyAxes()) ||
      newSharding == oldTensorSharding) {
    // This means no update can be done to the sharding.
    // TODO(tomnatan): find a way to warn about this silently.
    static llvm::once_flag flag;
    emitOpWarningOnce(flag, getOwningOp(modifiedValue),
                      "can't propagate sharding as strided view is needed");
    return false;
  }

  setTensorShardingCallback(newSharding);

  if (notifyOpModified) {
    notifyShardingModified(modifiedValue, *notifyOpModified);
  }

  // Set the sharding of all values in the same sharding group to be equivalent
  // (skipping the modified value which has already been updated).
  for (Value groupValue : shardingGroupMap.getGroupMembers(modifiedValue)) {
    if (groupValue == modifiedValue) {
      continue;
    }
    setSharding(groupValue, newSharding);
    if (notifyOpModified) {
      notifyShardingModified(groupValue, *notifyOpModified);
    }
  }

  return true;
}

// Updates the sharding of all tensors according to `tensorFactorShardings`.
//
// Skips tensors for which `updateTensor` is set to false.
//
// If an operand or result couldn't be updated to the corresponding sharding in
// `tensorFactorShardings`, e.g., if strided view is required, sets the
// respective bit in `updateTensor` or `updateResult` to false.
void updateTensorShardings(
    ValueRange tensors, ArrayRef<TensorShardingAttr> tensorShardings,
    SetShardingPerTensorCallback setTensorShardingCallback,
    ArrayRef<TensorFactorShardings> tensorFactorShardings,
    ArrayRef<TensorMappingAttr> tensorMappings, ArrayRef<int64_t> factorSizes,
    BitVector& updateTensor, StringRef meshName, MeshAttr mesh,
    const ShardingGroupMap& shardingGroupMap,
    std::optional<NotifyOpModifiedCallback> notifyOpModified) {
  for (int64_t index : updateTensor.set_bits()) {
    if (!updateTensorSharding(
            tensorShardings[index],
            std::bind(setTensorShardingCallback, std::placeholders::_1, index),
            tensorFactorShardings[index], tensorMappings[index], factorSizes,
            meshName, mesh, getShardableValue(tensors[index]), shardingGroupMap,
            notifyOpModified)) {
      updateTensor.reset(index);
    }
  }
}

// Same as the overload above, except operates on both operands and results.
void updateTensorShardings(
    ValueRange operands, ValueRange results,
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultShardings,
    SetShardingPerTensorCallback setOperandShardingCallback,
    SetShardingPerTensorCallback setResultShardingCallback,
    OpShardingRuleAttr shardingRule,
    const ShardingProjection& shardingProjection, BitVector& updateOperand,
    BitVector& updateResult, StringRef meshName, MeshAttr mesh,
    const ShardingGroupMap& shardingGroupMap,
    std::optional<NotifyOpModifiedCallback> notifyOpModified) {
  updateTensorShardings(operands, operandShardings, setOperandShardingCallback,
                        shardingProjection.getOperands(),
                        shardingRule.getOperandMappings(),
                        shardingRule.getFactorSizes(), updateOperand, meshName,
                        mesh, shardingGroupMap, notifyOpModified);
  updateTensorShardings(results, resultShardings, setResultShardingCallback,
                        shardingProjection.getResults(),
                        shardingRule.getResultMappings(),
                        shardingRule.getFactorSizes(), updateResult, meshName,
                        mesh, shardingGroupMap, notifyOpModified);
}

// Propagates tensor shardings of the given `operands` and `results` according
// to `shardingRule`.
//
// NOTE: the `operands`/`results` can be any sort of ValueRange associated to
// the Operation. For example, for CaseOp, an op with no operands, it's called
// with the return values of each branch/region.
LogicalResult propagateTensorShardings(
    ValueRange operands, ValueRange results,
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultShardings,
    SetShardingPerTensorCallback setOperandShardingCallback,
    SetShardingPerTensorCallback setResultShardingCallback,
    OpShardingRuleAttr shardingRule, PropagationDirection direction,
    const FactorPropagation& factorPropagation,
    const ShardingGroupMap& shardingGroupMap, bool conservativePropagation,
    Operation* op, const SymbolTable& symbolTable, PatternRewriter* rewriter) {
  std::optional<StringRef> meshName =
      getCommonMeshName(operandShardings, resultShardings, symbolTable);
  if (!meshName.has_value()) {
    // This means none of the operands or results have a sharding attribute or
    // the sharding attributes use different meshes.
    if (rewriter) {
      return rewriter->notifyMatchFailure(
          op, [](Diagnostic& diag) { diag << "no common mesh name found"; });
    }
    return failure();
  }
  MeshAttr mesh = getMeshAttr(op, meshName.value());
  assert(mesh && "unknown mesh");

  ShardingProjection shardingProjection = ShardingProjection::build(
      operandShardings, resultShardings, shardingRule, mesh);

  auto [updateOperand, updateResult] =
      factorPropagation.propagateFactorShardings(
          shardingProjection, direction, shardingRule.getFactorSizes(), mesh,
          op, conservativePropagation);

  // We need to update the tensor sharding attributes explicitly, as we have
  // been modifying our internal `shardingProjection` so far.
  std::optional<NotifyOpModifiedCallback> notifyOpModified = std::nullopt;
  if (rewriter) {
    notifyOpModified = [op, rewriter](Operation* modifiedOp) {
      // We don't want to add `op` itself back to the worklist since we have
      // just propagated through it, i.e., applying this method again on the
      // same op, without additional sharding changes, wouldn't do anything
      // other than redundant work.
      if (modifiedOp != op) {
        rewriter->modifyOpInPlace(modifiedOp, []() {});
      }
    };
  }

  op->getContext()->executeAction<SourceShardingAction>(
      [&, &updateOperand = updateOperand, &updateResult = updateResult]() {
        updateTensorShardings(
            operands, results, operandShardings, resultShardings,
            setOperandShardingCallback, setResultShardingCallback, shardingRule,
            shardingProjection, updateOperand, updateResult, meshName.value(),
            mesh, shardingGroupMap, notifyOpModified);
      },
      /*IRUnits=*/{op}, operands, results, operandShardings, resultShardings);

  bool anyUpdated = updateOperand.any() || updateResult.any();
  if (rewriter && !anyUpdated) {
    return rewriter->notifyMatchFailure(op, [](Diagnostic& diag) {
      diag << "Couldn't update any of the factor shardings";
    });
  }
  return success(anyUpdated);
}

// Same as the overload above, except there is a single operand and result.
LogicalResult propagateTensorShardings(
    Value operand, Value result, TensorShardingAttr operandSharding,
    TensorShardingAttr resultsSharding,
    SetTensorShardingCallback setOperandShardingCallback,
    SetTensorShardingCallback setResultShardingCallback,
    OpShardingRuleAttr shardingRule, Operation* op,
    const SymbolTable& symbolTable, PatternRewriter* rewriter,
    const FactorPropagation& factorPropagation,
    const ShardingGroupMap& shardingGroupMap,
    PropagationDirection direction = PropagationDirection::BOTH,
    bool conservativePropagation = false) {
  return propagateTensorShardings(
      operand, result, operandSharding, resultsSharding,
      [&](TensorShardingAttr sharding, int64_t) {
        setOperandShardingCallback(sharding);
      },
      [&](TensorShardingAttr sharding, int64_t) {
        setResultShardingCallback(sharding);
      },
      shardingRule, direction, factorPropagation, shardingGroupMap,
      conservativePropagation, op, symbolTable, rewriter);
}

// Same as the overload above, except the operand and result shardings are
// extracted using `getSharding` and set using `setSharding`.
LogicalResult propagateTensorShardings(
    ValueRange operands, ValueRange results, OpShardingRuleAttr shardingRule,
    Operation* op, const SymbolTable& symbolTable, PatternRewriter& rewriter,
    const FactorPropagation& factorPropagation,
    const ShardingGroupMap& shardingGroupMap,
    PropagationDirection direction = PropagationDirection::BOTH,
    bool conservativePropagation = false) {
  return propagateTensorShardings(
      operands, results, getShardings(operands), getShardings(results),
      [&](TensorShardingAttr sharding, int64_t index) {
        setSharding(operands[index], sharding);
      },
      [&](TensorShardingAttr sharding, int64_t index) {
        setSharding(results[index], sharding);
      },
      shardingRule, direction, factorPropagation, shardingGroupMap,
      conservativePropagation, op, symbolTable, &rewriter);
}

// Propagates the shardings between the operands of the `funcOp`'s terminator
// and the `funcOp`'s result type attrs.
LogicalResult propagateFuncResults(FuncOp funcOp,
                                   const SymbolTable& symbolTable,
                                   const FactorPropagation& factorPropagation,
                                   const ShardingGroupMap& shardingGroupMap) {
  for (OpOperand& returnOperand : getBodyTerminatorOpOperands(funcOp)) {
    Value returnValue = returnOperand.get();
    auto tensorType = dynCastStaticShapedType(returnValue.getType());
    if (!tensorType) {
      // Skip non-static-shaped tensors, e.g., tokens.
      continue;
    }
    int64_t resNum = returnOperand.getOperandNumber();
    // NOTE: we void the returned `LogicalResult` since function updates aren't
    // done through a rewriter, can ignore whether operands/results were
    // updated.
    (void)propagateTensorShardings(
        // The operand/result function arguments are used to:
        // - invoke the rewriter (if specified) that a value was updated. But
        //   a rewriter isn't used here.
        // - log warnings on the defining op. In this case it would either be on
        //   the defining op of `returnValue` or `funcOp` if it's a function
        //   argument. Here it will be okay to log the warning on the defining
        //   op of `returnValue`.
        // As such, we pass `returnValue` as both the operand and result.
        returnValue, returnValue, getSharding(returnValue),
        getFuncResultSharding(funcOp, resNum),
        [&](TensorShardingAttr sharding) {
          setSharding(returnValue, sharding);
        },
        [&](TensorShardingAttr sharding) {
          setFuncResultSharding(funcOp, resNum, sharding);
        },
        // Treat the sharding data flow b/w the `funcOp` terminator and func
        // result attrs as an identity op. Create an equivalent sharding
        // rule.
        createIdentityShardingRule(tensorType), funcOp, symbolTable,
        /*rewriter=*/nullptr, factorPropagation, shardingGroupMap);
  }
  return success();
}

// Overload of `propagateFuncResults` to propagate operand/result shardings of
// every `FuncOp` in `moduleOp`.
LogicalResult propagateFuncResults(ModuleOp moduleOp,
                                   const SymbolTable& symbolTable,
                                   const FactorPropagation& factorPropagation,
                                   const ShardingGroupMap& shardingGroupMap) {
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (failed(propagateFuncResults(funcOp, symbolTable, factorPropagation,
                                    shardingGroupMap))) {
      return failure();
    }
  }
  return success();
}

// Propagates the sharding of an operation (between operands and results) that
// has a registered or custom `OpShardingRuleAttr`.
class PropagateRegisteredOp : public RewritePattern {
 public:
  explicit PropagateRegisteredOp(
      MLIRContext* context, const SymbolTable& symbolTable,
      GetDirectionToPropagateFn getDirectionToPropagate,
      bool conservativePropagation, const FactorPropagation& factorPropagation,
      const ShardingGroupMap& shardingGroupMap)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        symbolTable(symbolTable),
        getDirectionToPropagate(getDirectionToPropagate),
        conservativePropagation(conservativePropagation),
        factorPropagation(factorPropagation),
        shardingGroupMap(shardingGroupMap) {}

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

    return propagateTensorShardings(op->getOperands(), op->getResults(),
                                    shardingRule, op, symbolTable, rewriter,
                                    factorPropagation, shardingGroupMap,
                                    direction, conservativePropagation);
  }

 private:
  const SymbolTable& symbolTable;
  GetDirectionToPropagateFn getDirectionToPropagate;
  bool conservativePropagation;
  const FactorPropagation& factorPropagation;
  const ShardingGroupMap& shardingGroupMap;
};

// Propagates shardings between the sources and targets of an
// `sdy.data_flow_edge`.
//
// The `sdy.data_flow_edge` holds the updateable sharding of all targets.
class PropagateDataFlowEdgeOp : public OpRewritePattern<DataFlowEdgeOp> {
 public:
  explicit PropagateDataFlowEdgeOp(MLIRContext* context,
                                   const SymbolTable& symbolTable,
                                   const FactorPropagation& factorPropagation,
                                   const ShardingGroupMap& shardingGroupMap)
      : OpRewritePattern<DataFlowEdgeOp>(context),
        symbolTable(symbolTable),
        factorPropagation(factorPropagation),
        shardingGroupMap(shardingGroupMap) {}

  LogicalResult matchAndRewrite(DataFlowEdgeOp dataFlowEdgeOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<Value> sources = getDataFlowSources(dataFlowEdgeOp);
    // The sharding of `dataFlowEdgeOp.getResult()` is the sharding of all
    // targets.
    return propagateTensorShardings(
        sources, dataFlowEdgeOp.getResult(), getShardings(sources),
        transformTargetSharding(
            dataFlowEdgeOp, dataFlowEdgeOp.getShardingAttr(),
            DataFlowShardingTransformType::kBeforeEdgePropagation),
        [&](TensorShardingAttr sharding, int64_t index) {
          setSharding(sources[index], sharding);
        },
        [&](TensorShardingAttr sharding, int64_t _) {
          dataFlowEdgeOp.setShardingAttr(transformTargetSharding(
              dataFlowEdgeOp, sharding,
              DataFlowShardingTransformType::kAfterEdgePropagation));
        },
        createIdentityShardingRule(cast<ShapedType>(dataFlowEdgeOp.getType()),
                                   sources.size()),
        PropagationDirection::BOTH, factorPropagation, shardingGroupMap,
        /*conservativePropagation=*/false, dataFlowEdgeOp, symbolTable,
        &rewriter);
  }

 private:
  const SymbolTable& symbolTable;
  const FactorPropagation& factorPropagation;
  const ShardingGroupMap& shardingGroupMap;
};

// Propagates through a `PropagationBarrierOp` accounting for the direction in
// which it blocks propagation.
class PropagatePropagationBarrier
    : public OpRewritePattern<PropagationBarrierOp> {
 public:
  explicit PropagatePropagationBarrier(
      MLIRContext* context, const SymbolTable& symbolTable,
      const FactorPropagation& factorPropagation,
      const ShardingGroupMap& shardingGroupMap)
      : OpRewritePattern<PropagationBarrierOp>(context),
        symbolTable(symbolTable),
        factorPropagation(factorPropagation),
        shardingGroupMap(shardingGroupMap) {}

  LogicalResult matchAndRewrite(PropagationBarrierOp propagationBarrierOp,
                                PatternRewriter& rewriter) const override {
    return propagateTensorShardings(
        propagationBarrierOp.getInput(), propagationBarrierOp.getResult(),
        createIdentityShardingRule(
            cast<RankedTensorType>(propagationBarrierOp.getType())),
        propagationBarrierOp, symbolTable, rewriter, factorPropagation,
        shardingGroupMap, propagationBarrierOp.getAllowedDirection());
  }

 private:
  const SymbolTable& symbolTable;
  const FactorPropagation& factorPropagation;
  const ShardingGroupMap& shardingGroupMap;
};

// The basic propagation pass that uses the default implementation of
// `BasicPropagationPassImpl`.
struct BasicPropagationPass
    : public impl::BasicPropagationPassBase<BasicPropagationPass> {
  using BasicPropagationPassBase::BasicPropagationPassBase;

  explicit BasicPropagationPass(const PropagationOptions& options) {
    setPropagationOptions(options);
  }
};

// Verifies that all shapes are static and there aren't any tuple types.
bool allValidShapes(ModuleOp moduleOp) {
  return !moduleOp
              .walk([](Operation* op) {
                for (Type type : op->getResultTypes()) {
                  if (auto tensorType = dyn_cast<ShapedType>(type);
                      tensorType && !tensorType.hasStaticShape()) {
                    op->emitError(
                        "Shardy propagation only supports ranked tensors with "
                        "a static shape. type: ")
                        << tensorType;
                    return WalkResult::interrupt();
                  }
                  if (auto tupleType = dyn_cast<TupleType>(type)) {
                    op->emitError(
                        "Shardy propagation doesn't support tuples: ")
                        << tupleType;
                    return WalkResult::interrupt();
                  }
                }
                return WalkResult::advance();
              })
              .wasInterrupted();
}

}  // namespace

PropagationDirection propagateAny(Operation*) {
  return PropagationDirection::BOTH;
}

LogicalResult BasicPropagationPassImpl::propagate(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    const ShardingGroupMap& shardingGroupMap,
    const FactorPropagation& factorPropagation,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  // Pushes any shardings that exist on the `funcOp` result type attrs to the
  // corresponding values returned in the terminator of the body of `funcOp`.
  if (failed(propagateFuncResults(moduleOp, symbolTable, factorPropagation,
                                  shardingGroupMap))) {
    return failure();
  }
  MLIRContext* context = moduleOp.getContext();
  RewritePatternSet patterns(context);
  patterns.add<PropagateDataFlowEdgeOp, PropagatePropagationBarrier>(
      context, symbolTable, factorPropagation, shardingGroupMap);
  patterns.add<PropagateRegisteredOp>(
      context, symbolTable, getDirectionToPropagate, conservativePropagation,
      factorPropagation, shardingGroupMap);
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
  if (failed(propagateFuncResults(moduleOp, symbolTable, factorPropagation,
                                  shardingGroupMap))) {
    return failure();
  }
  return success();
}

LogicalResult BasicPropagationPassImpl::propagate(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    const ShardingGroupMap& shardingGroupMap,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  return propagate(moduleOp, symbolTable, shardingGroupMap,
                   basicFactorPropagation, getDirectionToPropagate);
}

void BasicPropagationPassImpl::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext& context = getContext();

  // Prepare debugging handler for sharding origins and edge sources.
  ShardingDebugMappings mappings(debugShardingOrigins, debugEdgeSourceSharding);
  SourceShardingHandler handler(&mappings);
  handler.prepareHandler(moduleOp);

  SymbolTable symbolTable(moduleOp);

  if (!allValidShapes(moduleOp)) {
    return signalPassFailure();
  }

  // Build sharding group mappings to link values to other members within their
  // group. These maps are passed through the propagation methods so that
  // `updateTensorShardings` can enforce the sharding group constraints.
  ShardingGroupMap shardingGroupMap(moduleOp);
  if (failed(propagate(moduleOp, symbolTable, shardingGroupMap))) {
    signalPassFailure();
    return;
  }
  if (!keepShardingRules) {
    removeShardingRules(moduleOp);
  }

  context.registerActionHandler(nullptr);
  handler.saveOnModule(moduleOp);

  saveModuleOp(moduleOp, dumpDirectory, "sdy_module_after_propagation");
}

void BasicPropagationPassImpl::setPropagationOptions(
    const PropagationOptions& options) {
  keepShardingRules = options.keepShardingRules;
  dumpDirectory = options.dumpDirectory.str();
  conservativePropagation = options.conservativePropagation;
  debugShardingOrigins = options.debugShardingOrigins;
  debugEdgeSourceSharding = options.debugEdgeSourceSharding;
}

std::unique_ptr<Pass> createBasicPropagationPass(
    const PropagationOptions& options) {
  return std::make_unique<BasicPropagationPass>(options);
}

}  // namespace sdy
}  // namespace mlir
