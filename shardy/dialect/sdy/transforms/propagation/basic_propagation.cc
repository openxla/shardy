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
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/debugging/source_sharding.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
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

    if (auto dataFlowEdge = DataFlowEdgeOp::lookup(use)) {
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
    for (Value nonEdgeOwnerTarget : dataFlowEdge.getNonOwnerTargets()) {
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

// Struct to hold common parameters for sharding propagation.
struct PropagationSharedParams {
  const ShardingGroupMap& shardingGroupMap;
  StringRef meshName;
  MeshAttr mesh;
  std::optional<NotifyOpModifiedCallback> notifyOpModified;
};

struct PropagationTensorParams {
  ValueRange tensors;
  ArrayRef<TensorShardingAttr> shardings;
  SetShardingPerTensorCallback setShardingCallback;

  PropagationTensorParams(ValueRange tensors,
                          ArrayRef<TensorShardingAttr> shardings,
                          SetShardingPerTensorCallback setShardingCallback)
      : tensors(tensors),
        shardings(shardings),
        setShardingCallback(setShardingCallback) {}
};

// Update the sharding of `value` to the sharding in `tensorFactorShardings`.
//
// Returns true if it's possible to update the sharding, i.e., if strided view
// isn't needed and all non-minor-most factors are divisible by sharding axes.
bool updateTensorSharding(Value modifiedValue,
                          TensorShardingAttr oldTensorSharding,
                          SetTensorShardingCallback setTensorShardingCallback,
                          const TensorFactorShardings& tensorFactorShardings,
                          TensorMappingAttr tensorMapping,
                          ArrayRef<int64_t> factorSizes,
                          const PropagationSharedParams& params) {
  // We can assume `modifiedValue` exists since we are updating its sharding.
  assert(modifiedValue && "modified value should exist");
  TensorShardingAttr newSharding =
      tensorFactorShardings.createTensorShardingAttr(
          params.mesh.getContext(), tensorMapping, factorSizes, params.meshName,
          params.mesh);
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

  if (params.notifyOpModified) {
    notifyShardingModified(modifiedValue, *params.notifyOpModified);
  }

  // Set the sharding of all values in the same sharding group to be equivalent
  // (skipping the modified value which has already been updated).
  for (Value groupValue :
       params.shardingGroupMap.getGroupMembers(modifiedValue)) {
    if (groupValue == modifiedValue) {
      continue;
    }
    setSharding(groupValue, newSharding);
    if (params.notifyOpModified) {
      notifyShardingModified(groupValue, *params.notifyOpModified);
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
    const PropagationTensorParams& tensorParams,
    ArrayRef<TensorFactorShardings> tensorFactorShardings,
    ArrayRef<TensorMappingAttr> tensorMappings, ArrayRef<int64_t> factorSizes,
    BitVector& updateTensor, const PropagationSharedParams& params) {
  for (int64_t index : updateTensor.set_bits()) {
    if (!updateTensorSharding(getShardableValue(tensorParams.tensors[index]),
                              tensorParams.shardings[index],
                              std::bind(tensorParams.setShardingCallback,
                                        std::placeholders::_1, index),
                              tensorFactorShardings[index],
                              tensorMappings[index], factorSizes, params)) {
      updateTensor.reset(index);
    }
  }
}

// Same as the overload above, except operates on both operands and results.
void updateTensorShardings(const PropagationTensorParams& operandsParams,
                           const PropagationTensorParams& resultsParams,
                           OpShardingRuleAttr shardingRule,
                           const ShardingProjection& shardingProjection,
                           BitVector& updateOperand, BitVector& updateResult,
                           const PropagationSharedParams& params) {
  updateTensorShardings(operandsParams, shardingProjection.getOperands(),
                        shardingRule.getOperandMappings(),
                        shardingRule.getFactorSizes(), updateOperand, params);
  updateTensorShardings(resultsParams, shardingProjection.getResults(),
                        shardingRule.getResultMappings(),
                        shardingRule.getFactorSizes(), updateResult, params);
}

// Propagates tensor shardings of the given `operands` and `results` according
// to `shardingRule`.
//
// NOTE: the `operands`/`results` can be any sort of ValueRange associated to
// the Operation. For example, for CaseOp, an op with no operands, it's called
// with the return values of each branch/region.
LogicalResult propagateTensorShardings(
    const PropagationTensorParams& operandsParams,
    const PropagationTensorParams& resultsParams,
    OpShardingRuleAttr shardingRule,
    PropagationDirectionAlongFactor directionAlongFactor,
    const FactorPropagation& factorPropagation, bool conservativePropagation,
    Operation* op, const SymbolTable& symbolTable, PatternRewriter* rewriter,
    ShardingGroupMap shardingGroupMap) {
  std::optional<StringRef> meshName = getCommonMeshName(
      operandsParams.shardings, resultsParams.shardings, symbolTable);

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

  ShardingProjection shardingProjection = ShardingProjection::build(
      operandsParams.shardings, resultsParams.shardings, shardingRule, mesh);
  bool anyUpdated = false;
  auto updateShardings = [&]() {
    auto [updateOperand, updateResult] =
        factorPropagation.propagateFactorShardings(
            shardingProjection, directionAlongFactor,
            shardingRule.getFactorSizes(), mesh, op, conservativePropagation);
    PropagationSharedParams params{shardingGroupMap, meshName.value(), mesh,
                                   notifyOpModified};

    updateTensorShardings(operandsParams, resultsParams, shardingRule,
                          shardingProjection, updateOperand, updateResult,
                          params);

    anyUpdated = updateOperand.any() || updateResult.any();
  };

  MLIRContext* context = op->getContext();
  if (context->hasActionHandler()) {
    context->executeAction<SourceShardingAction>(
        updateShardings,
        /*IRUnits=*/{op}, op, operandsParams.tensors, resultsParams.tensors,
        mesh, shardingRule, shardingProjection, anyUpdated);
  } else {
    updateShardings();
  }

  if (rewriter && !anyUpdated) {
    return rewriter->notifyMatchFailure(op, [](Diagnostic& diag) {
      diag << "Couldn't update any of the factor shardings";
    });
  }
  return success(anyUpdated);
}

// Same as the overload above, except the operand and result shardings are
// extracted using `getSharding` and set using `setSharding`.
LogicalResult propagateTensorShardings(
    ValueRange operands, ValueRange results, OpShardingRuleAttr shardingRule,
    Operation* op, const SymbolTable& symbolTable, PatternRewriter& rewriter,
    PropagationDirectionAlongFactor directionAlongFactor,
    const FactorPropagation& factorPropagation,
    const ShardingGroupMap& shardingGroupMap,
    bool conservativePropagation = false) {
  SmallVector<TensorShardingAttr> operandsShardings = getShardings(operands);
  SmallVector<TensorShardingAttr> resultsShardings = getShardings(results);
  PropagationTensorParams operandsParams = PropagationTensorParams(
      /*tensors=*/operands,
      /*shardings=*/operandsShardings,
      /*setShardingCallback=*/[&](TensorShardingAttr sharding, int64_t index) {
        setSharding(operands[index], sharding);
      });
  PropagationTensorParams resultsParams = PropagationTensorParams(
      /*tensors=*/results,
      /*shardings=*/resultsShardings,
      /*setShardingCallback=*/[&](TensorShardingAttr sharding, int64_t index) {
        setSharding(results[index], sharding);
      });

  return propagateTensorShardings(operandsParams, resultsParams, shardingRule,
                                  directionAlongFactor, factorPropagation,
                                  conservativePropagation, op, symbolTable,
                                  &rewriter, shardingGroupMap);
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
    // The operand/result function arguments are used to:
    // - invoke the rewriter (if specified) that a value was updated. But
    //   a rewriter isn't used here.
    // - log warnings on the defining op. In this case it would either be on
    //   the defining op of `returnValue` or `funcOp` if it's a function
    //   argument. Here it will be okay to log the warning on the defining
    //   op of `returnValue`.
    // As such, we pass `returnValue` as both the operand and result.
    TensorShardingAttr operandShardingRef = getSharding(returnValue);
    TensorShardingAttr resultsShardingRef =
        getFuncResultSharding(funcOp, resNum);
    PropagationTensorParams operandsParams = PropagationTensorParams(
        /*tensors=*/returnValue,
        /*shardings=*/operandShardingRef,
        /*setShardingCallback=*/[&](TensorShardingAttr sharding, int64_t) {
          setSharding(returnValue, sharding);
        });
    PropagationTensorParams resultsParams = PropagationTensorParams(
        /*tensors=*/returnValue,
        /*shardings=*/resultsShardingRef,
        /*setShardingCallback=*/[&](TensorShardingAttr sharding, int64_t) {
          setFuncResultSharding(funcOp, resNum, sharding);
        });

    (void)propagateTensorShardings(
        operandsParams, resultsParams,
        // Treat the sharding data flow b/w the `funcOp` terminator and func
        // result attrs as an identity op. Create an equivalent sharding rule.
        createIdentityShardingRule(tensorType),
        std::bind(propagateAny, funcOp, std::placeholders::_1),
        factorPropagation,
        /*conservativePropagation=*/false, funcOp, symbolTable,
        /*rewriter=*/nullptr, shardingGroupMap);
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
      const FactorPropagation& factorPropagation, bool conservativePropagation,
      const ShardingGroupMap& shardingGroupMap)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        symbolTable(symbolTable),
        getDirectionToPropagate(getDirectionToPropagate),
        factorPropagation(factorPropagation),
        conservativePropagation(conservativePropagation),
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

    PropagationDirectionAlongFactor directionAlongFactor =
        std::bind(getDirectionToPropagate, op, std::placeholders::_1);
    return propagateTensorShardings(op->getOperands(), op->getResults(),
                                    shardingRule, op, symbolTable, rewriter,
                                    directionAlongFactor, factorPropagation,
                                    shardingGroupMap, conservativePropagation);
  }

 private:
  const SymbolTable& symbolTable;
  GetDirectionToPropagateFn getDirectionToPropagate;
  const FactorPropagation& factorPropagation;
  bool conservativePropagation;
  const ShardingGroupMap& shardingGroupMap;
};

// Propagates shardings between the sources and targets of an
// `sdy.data_flow_edge`.
//
// The `sdy.data_flow_edge` holds the updateable sharding of all targets.
class PropagateDataFlowEdgeOp : public OpRewritePattern<DataFlowEdgeOp> {
 public:
  explicit PropagateDataFlowEdgeOp(
      MLIRContext* context, const SymbolTable& symbolTable,
      GetDirectionToPropagateFn getDirectionToPropagate,
      const FactorPropagation& factorPropagation,
      const ShardingGroupMap& shardingGroupMap)
      : OpRewritePattern<DataFlowEdgeOp>(context),
        symbolTable(symbolTable),
        getDirectionToPropagate(getDirectionToPropagate),
        factorPropagation(factorPropagation),
        shardingGroupMap(shardingGroupMap) {}

  LogicalResult matchAndRewrite(DataFlowEdgeOp dataFlowEdgeOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<Value> sources = dataFlowEdgeOp.getSources();
    SmallVector<TensorShardingAttr> operandShardingRef = getShardings(sources);
    PropagationTensorParams operandsParams = PropagationTensorParams(
        /*tensors=*/sources,
        /*shardings=*/operandShardingRef,
        /*setShardingCallback=*/
        [&sources](TensorShardingAttr sharding, int64_t index) {
          setSharding(sources[index], sharding);
        });

    Value result = dataFlowEdgeOp.getResult();
    // The sharding of `result` is the sharding of all targets.
    TensorShardingAttr resultsShardingRef =
        dataFlowEdgeOp.transformTargetSharding(
            dataFlowEdgeOp.getShardingAttr(),
            DataFlowShardingTransformType::kBeforeEdgePropagation);
    PropagationTensorParams resultsParams = PropagationTensorParams(
        /*tensors=*/result,
        /*shardings=*/resultsShardingRef,
        /*setShardingCallback=*/
        [&dataFlowEdgeOp](TensorShardingAttr sharding, int64_t) {
          dataFlowEdgeOp.setShardingAttr(dataFlowEdgeOp.transformTargetSharding(
              sharding, DataFlowShardingTransformType::kAfterEdgePropagation));
        });

    PropagationDirectionAlongFactor directionAlongFactor = std::bind(
        getDirectionToPropagate, dataFlowEdgeOp, std::placeholders::_1);
    return propagateTensorShardings(
        operandsParams, resultsParams,
        createIdentityShardingRule(cast<ShapedType>(dataFlowEdgeOp.getType()),
                                   sources.size()),
        directionAlongFactor, factorPropagation,
        /*conservativePropagation=*/false, dataFlowEdgeOp, symbolTable,
        &rewriter, shardingGroupMap);
  }

 private:
  const SymbolTable& symbolTable;
  GetDirectionToPropagateFn getDirectionToPropagate;
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
        propagationBarrierOp, symbolTable, rewriter,
        [&](int64_t) { return propagationBarrierOp.getAllowedDirection(); },
        factorPropagation, shardingGroupMap);
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

PropagationDirection propagateAny(Operation*, int64_t) {
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
  patterns.add<PropagatePropagationBarrier>(
      context, symbolTable, factorPropagation, shardingGroupMap);
  patterns.add<PropagateDataFlowEdgeOp>(context, symbolTable,
                                        getDirectionToPropagate,
                                        factorPropagation, shardingGroupMap);
  patterns.add<PropagateRegisteredOp>(
      context, symbolTable, getDirectionToPropagate, factorPropagation,
      conservativePropagation, shardingGroupMap);
  // We only need a single iteration (and another to confirm convergence), since
  // we make sure ops whose sharding changes are added back to the worklist.
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  config.fold = false;
  config.cseConstants = false;
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns), config))) {
    // We should always converge in 2 iterations, if we don't, something is
    // wrong.
    moduleOp->emitError("Failed to converge after ")
        << config.maxIterations
        << " iterations. please contact the Shardy team.";
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
  ShardingDebugMappings mappings(debugShardingOrigins,
                                 debugPropagationEdgeSharding);
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
  debugPropagationEdgeSharding = options.debugPropagationEdgeSharding;
}

std::unique_ptr<Pass> createBasicPropagationPass(
    const PropagationOptions& options) {
  return std::make_unique<BasicPropagationPass>(options);
}

}  // namespace sdy
}  // namespace mlir
