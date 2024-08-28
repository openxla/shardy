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

#include "shardy/dialect/sdy/transforms/propagation/sharding_propagation_context.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/data_flow_utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_propagation_context.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir {
namespace sdy {

using func::FuncOp;

// Sets the sharding of a tensor at a given index to the given
// `TensorShardingAttr`.
using SetShardingPerTensorCallback =
    std::function<void(TensorShardingAttr, int64_t)>;

// Sets the sharding of a tensor to the given `TensorShardingAttr`.
using SetTensorShardingCallback = std::function<void(TensorShardingAttr)>;

using NotifyOpModifiedCallback = std::function<void(Operation*)>;

namespace {

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
    forEachNonRootDataFlowTarget(dataFlowEdge, [&](Value value) {
      notifyUsersModified(value, notifyOpModified);
    });
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
    std::optional<NotifyOpModifiedCallback> notifyOpModified) {
  for (int64_t index : updateTensor.set_bits()) {
    if (!updateTensorSharding(
            tensorShardings[index],
            std::bind(setTensorShardingCallback, std::placeholders::_1, index),
            tensorFactorShardings[index], tensorMappings[index], factorSizes,
            meshName, mesh, getShardableValue(tensors[index]),
            notifyOpModified)) {
      updateTensor.reset(index);
    }
  }
}

// Same as the overload above, except operates on both operands and results.
void updateTensorShardings(
    ValueRange operands, ValueRange results,
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultsShardings,
    SetShardingPerTensorCallback setOperandShardingCallback,
    SetShardingPerTensorCallback setResultShardingCallback,
    OpShardingRuleAttr shardingRule,
    const ShardingProjection& shardingProjection, BitVector& updateOperand,
    BitVector& updateResult, StringRef meshName, MeshAttr mesh,
    std::optional<NotifyOpModifiedCallback> notifyOpModified) {
  updateTensorShardings(operands, operandShardings, setOperandShardingCallback,
                        shardingProjection.getOperands(),
                        shardingRule.getOperandMappings(),
                        shardingRule.getFactorSizes(), updateOperand, meshName,
                        mesh, notifyOpModified);
  updateTensorShardings(results, resultsShardings, setResultShardingCallback,
                        shardingProjection.getResults(),
                        shardingRule.getResultMappings(),
                        shardingRule.getFactorSizes(), updateResult, meshName,
                        mesh, notifyOpModified);
}

}  // namespace

LogicalResult ShardingPropagationContext::propagateFuncResults(
    FuncOp funcOp, const FactorPropagation& factorPropagation) const {
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
    (void)propagateSingleTensorSharding(
        // The operand/result function arguments are used to:
        // - invoke the rewriter (if specified) that a value was updated. But
        //   a rewriter isn't used here.
        // - log warnings on the defining op. In this case it would either be on
        //   the defining op of `returnValue` or `funcOp` if it's a function
        //   argument. Here it will be okay to log the warning on the defining
        //   op of `returnValue`.
        // As such, we pass `returnValue` as both the operand and result.
        returnValue, returnValue,
        getSharding(returnValue),  // valueToShardingGroup
        funcOp.getResultAttrOfType<TensorShardingAttr>(resNum, kShardingAttr),
        [&](TensorShardingAttr sharding) {
          setSharding(returnValue, sharding);  // #shardingGroupToValues
        },
        [&](TensorShardingAttr sharding) {
          funcOp.setResultAttr(resNum, kShardingAttr, sharding);
        },
        // Treat the sharding data flow b/w the `funcOp` terminator and func
        // result attrs as an identity op. Create an equivalent sharding
        // rule.
        createIdentityShardingRule(tensorType), funcOp, /*rewriter=*/nullptr,
        factorPropagation);
  }
  return success();
}

LogicalResult ShardingPropagationContext::propagateAllFuncResultsInModule(
    ModuleOp moduleOp, const FactorPropagation& factorPropagation) const {
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (failed(propagateFuncResults(funcOp, factorPropagation))) {
      return failure();
    }
  }
  return success();
}

LogicalResult ShardingPropagationContext::propagateTensorShardings(
    ValueRange operands, ValueRange results,
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultsShardings,
    SetShardingPerTensorCallback setOperandShardingCallback,
    SetShardingPerTensorCallback setResultShardingCallback,
    OpShardingRuleAttr shardingRule, PropagationDirection direction,
    const FactorPropagation& factorPropagation, bool conservativePropagation,
    Operation* op, PatternRewriter* rewriter) const {
  std::optional<StringRef> meshName =
      getCommonMeshName(operandShardings, resultsShardings);
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
      operandShardings, resultsShardings, shardingRule, mesh);

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
  updateTensorShardings(operands, results, operandShardings, resultsShardings,
                        setOperandShardingCallback, setResultShardingCallback,
                        shardingRule, shardingProjection, updateOperand,
                        updateResult, meshName.value(), mesh, notifyOpModified);

  bool anyUpdated = updateOperand.any() || updateResult.any();
  if (rewriter && !anyUpdated) {
    return rewriter->notifyMatchFailure(op, [](Diagnostic& diag) {
      diag << "Couldn't update any of the factor shardings";
    });
  }
  return success(anyUpdated);
}

LogicalResult ShardingPropagationContext::propagateSingleTensorSharding(
    Value operand, Value result, TensorShardingAttr operandSharding,
    TensorShardingAttr resultsSharding,
    SetTensorShardingCallback setOperandShardingCallback,
    SetTensorShardingCallback setResultShardingCallback,
    OpShardingRuleAttr shardingRule, Operation* op, PatternRewriter* rewriter,
    const FactorPropagation& factorPropagation, PropagationDirection direction,
    bool conservativePropagation) const {
  return propagateTensorShardings(
      operand, result, operandSharding, resultsSharding,
      [&](TensorShardingAttr sharding, int64_t) {
        setOperandShardingCallback(sharding);
      },
      [&](TensorShardingAttr sharding, int64_t) {
        setResultShardingCallback(sharding);
      },
      shardingRule, direction, factorPropagation, conservativePropagation, op,
      rewriter);
}

// Same as the overload above, except the operand and result shardings are
// extracted using `getSharding` and set using `setSharding`.
LogicalResult
ShardingPropagationContext::propagateTensorShardingsWithDefaultCallbacks(
    ValueRange operands, ValueRange results, OpShardingRuleAttr shardingRule,
    Operation* op, PatternRewriter& rewriter,
    const FactorPropagation& factorPropagation, PropagationDirection direction,
    bool conservativePropagation) const {
  return propagateTensorShardings(
      operands, results, getShardings(operands), getShardings(results),
      [&](TensorShardingAttr sharding, int64_t index) {
        setSharding(operands[index], sharding);
      },
      [&](TensorShardingAttr sharding, int64_t index) {
        setSharding(results[index], sharding);
      },
      shardingRule, direction, factorPropagation, conservativePropagation, op,
      &rewriter);
}

}  // namespace sdy
}  // namespace mlir
