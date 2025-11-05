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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>  // IWYU pragma: keep
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

void insertExplicitReshardsToTargetSharding(OpOperand& opOperand,
                                            TensorShardingAttr targetSharding,
                                            IRRewriter& rewriter,
                                            const SymbolTable& symbolTable,
                                            const bool insertAfterOperand,
                                            const bool onFullVersion) {
  Value operand = opOperand.get();
  TensorShardingAttr operandSharding = getSharding(operand);

  if (insertAfterOperand) {
    rewriter.setInsertionPointAfterValue(operand);
  }

  if (operandSharding) {
    // If `operandSharding` has unreduced axes, insert an all-reduce if any of
    // the axes isn't unreduced in the target sharding.
    operandSharding = insertAllReduceIfUnreducedToReplicated(
        opOperand, operandSharding, targetSharding, symbolTable, rewriter);
  }

  if (onFullVersion && !isEquivalent(operandSharding, targetSharding)) {
    operand = opOperand.get();
    auto reshardOp = ReshardOp::create(
        rewriter, operand.getLoc(), operand,
        targetSharding
            ? targetSharding
            // Since operand and target shardings are not equivalent and
            // `targetSharding` is empty, `operandSharding` is guaranteed to be
            // nonempty.
            : TensorShardingAttr::getFullyClosedLike(operandSharding));
    opOperand.set(reshardOp);
  }
}

void insertExplicitReshardsOnFuncReturn(Operation* op, func::FuncOp& funcOp,
                                        IRRewriter& rewriter,
                                        const SymbolTable& symbolTable,
                                        const bool onFullVersion) {
  rewriter.setInsertionPoint(op);
  for (const auto& [index, opOperand] : llvm::enumerate(op->getOpOperands())) {
    insertExplicitReshardsToTargetSharding(
        opOperand, /*targetSharding=*/getFuncResultSharding(funcOp, index),
        rewriter, symbolTable, /*insertAfterOperand=*/false, onFullVersion);
  }
}

void insertExplicitReshardsOnDataFlowOp(
    ShardableDataFlowOpInterface& op, IRRewriter& rewriter,
    const SymbolTable& symbolTable, const bool onFullVersion,
    const bool avoidReshardsOnNamedComputations) {
  if (isa<NamedComputationOp>(op) && avoidReshardsOnNamedComputations) {
    for (Value owner : op.getOpResultEdgeOwners()) {
      for (OpOperand* sourceOpOperand : op.getEdgeSources(owner)) {
        insertExplicitReshardsToTargetSharding(
            *sourceOpOperand,
            /*targetSharding=*/op.getEdgeOwnerSharding(owner), rewriter,
            symbolTable,
            /*insertAfterOperand=*/true, onFullVersion);
      }
    }
    return;
  }
  for (Value owner : llvm::concat<Value>(op.getOpResultEdgeOwners(),
                                         op.getBlockArgumentEdgeOwners())) {
    TensorShardingAttr ownerSharding = op.transformTargetSharding(
        owner, op.getEdgeOwnerSharding(owner),
        DataFlowShardingTransformType::kBeforeEdgePropagation);
    for (OpOperand* sourceOpOperand : op.getEdgeSources(owner)) {
      insertExplicitReshardsToTargetSharding(
          *sourceOpOperand,
          /*targetSharding=*/ownerSharding, rewriter, symbolTable,
          /*insertAfterOperand=*/true, onFullVersion);
    }
  }
}

// Reshard the result of a dot operation if all the following hold:
//
// 1. LHS and RHS have fully compatible shardings.
// 2. The result has exactly one dim (batching or non-contracting) that
//    conflicts with either LHS or RHS.
// 3. The result is smaller or equal in size to the conflicting operand.
// 4. The conflicting LHS/RHS sharding isn't empty, otherwise this is a common
//    reduce-scatter pattern.
// 5. The conflicting result sharding match the reduction axes in order (i.e.,
//    the axes that shard the contracting dimensions of LHS and RHS).
//
// In which case, we are getting a more sharded matmul and a reshard on a
// smaller tensor (result instead of LHS or RHS), while doing an all-reduce
// instead of a reduce-scatter on a bigger tensor.
//
// For example:
//
// ```mlir
// %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>}
// %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}
//
// %result = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
//  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} :
//  (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
//
// return %result : tensor<4x8xf32>
// ```
//
// ~>
//
// ```mlir
// %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>}
// %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}
//
// %result = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
//  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} :
//  (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
//
// %reshard = sdy.reshard %result <@mesh, [{}, {"x"}]>
//
// return %reshard  : tensor<4x8xf32>
// ```
template <class OpTy>
void processDot(OpTy op, ShardingProjection& shardingProjection,
                ArrayRef<TensorShardingAttr> outShardings, IRRewriter& rewriter,
                const SymbolTable& symbolTable, OpShardingRuleAttr shardingRule,
                const Mesh& mesh) {
  if (outShardings.empty()) {
    // Result doesn't have a sharding.
    return;
  }
  const TensorFactorShardings& lhsSharding = shardingProjection.getOperand(0);
  const TensorFactorShardings& rhsSharding = shardingProjection.getOperand(1);
  TensorFactorShardings& resultSharding =
      shardingProjection.getMutableResult(0);

  int64_t lhsSize = op.getLhs().getType().getNumElements();
  int64_t rhsSize = op.getRhs().getType().getNumElements();
  int64_t resultSize = op.getType().getNumElements();

  std::optional<int64_t> conflictingFactor;
  ArrayRef<AxisRefAttr> conflictingOperandAxes;
  SmallVector<AxisRefAttr> reductionAxes;
  SmallVector<AxisRefAttr> seenAxesAlongNonContractingDim;
  for (int64_t factorIndex = 0; factorIndex < shardingRule.getNumFactors();
       ++factorIndex) {
    std::optional<ArrayRef<AxisRefAttr>> lhsAxes =
        getFactorSharding(lhsSharding, factorIndex);
    std::optional<ArrayRef<AxisRefAttr>> rhsAxes =
        getFactorSharding(rhsSharding, factorIndex);
    ArrayRef<AxisRefAttr> operandAxes;
    int64_t maxOperandSize;

    if (lhsAxes && rhsAxes) {
      // Batching or contracting dim.
      operandAxes = *lhsAxes;
      maxOperandSize = std::max(lhsSize, rhsSize);
      if (lhsAxes != rhsAxes) {
        // Conflict between lhs and rhs.
        return;
      }
      if (shardingRule.isReductionFactor(factorIndex)) {
        // Contracting dim.
        reductionAxes.append(operandAxes.begin(), operandAxes.end());
        continue;
      }
    } else {
      // LHS or RHS non-contracting dim.
      operandAxes = lhsAxes ? *lhsAxes : *rhsAxes;
      maxOperandSize = lhsAxes ? lhsSize : rhsSize;
      for (AxisRefAttr axis : operandAxes) {
        if (llvm::any_of(seenAxesAlongNonContractingDim,
                         [&](AxisRefAttr a) { return a.overlaps(axis); })) {
          // Conflict between lhs and rhs non-contracting.
          return;
        }
        seenAxesAlongNonContractingDim.push_back(axis);
      }
    }

    // Safe to dereference since we skipped contracting dims above.
    if (*getFactorSharding(resultSharding, factorIndex) != operandAxes) {
      if (operandAxes.empty()) {
        // Conflicting LHS/RHS sharding is empty.
        return;
      }
      if (conflictingFactor) {
        // Multiple conflicting factors.
        return;
      }
      if (resultSize > maxOperandSize) {
        // Result is larger than operand with conflicting factor.
        return;
      }
      // Conflict between result and lhs/rhs.
      conflictingFactor = factorIndex;
      conflictingOperandAxes = operandAxes;
    }
  }

  if (!conflictingFactor) {
    // No conflicts.
    return;
  }

  SmallVector<AxisRefAttr>& resAxes =
      resultSharding.factorIndexToSharding[*conflictingFactor].axisRefs;
  if (resAxes.empty() || resAxes != reductionAxes) {
    // Result conflicting sharding doesn't match reduction axes.
    return;
  }

  // We can reshard the result to make it compatible with the sharding of the
  // LHS and RHS.
  resAxes.assign(conflictingOperandAxes.begin(), conflictingOperandAxes.end());
  setSharding(op.getResult(),
              resultSharding.createTensorShardingAttr(
                  op.getContext(), shardingRule.getResultMapping(0),
                  shardingRule.getFactorSizes(), mesh.name(), mesh.attr()));
  rewriter.setInsertionPointAfter(op);
  auto reshardOp = ReshardOp::create(rewriter, op.getLoc(), op.getResult(),
                                     outShardings.front());
  rewriter.replaceAllUsesExcept(op.getResult(), reshardOp, reshardOp);
}

Mesh getMeshOrDefault(TensorShardingAttr sharding,
                      const SymbolTable& symbolTable, const Mesh& defaultMesh) {
  if (!sharding) {
    return defaultMesh;
  }
  // NOTE: sharding always has a meshOrRef because it is a required parameter.
  return Mesh(sharding.getMesh(symbolTable),
              cast<FlatSymbolRefAttr>(sharding.getMeshOrRef()).getValue());
}

Mesh getMostCommonMesh(ArrayRef<TensorShardingAttr> inShardings,
                       ArrayRef<TensorShardingAttr> outShardings,
                       const SymbolTable& symbolTable,
                       const Mesh& defaultMesh) {
  int64_t maxMeshCount = 0;
  llvm::SmallDenseMap<StringRef, int64_t> meshCounts;
  Mesh mostCommonMesh = defaultMesh;
  for (const TensorShardingAttr sharding :
       llvm::concat<const TensorShardingAttr>(inShardings, outShardings)) {
    if (!isFullyReplicated(sharding)) {
      const Mesh meshOfSharding =
          getMeshOrDefault(sharding, symbolTable, defaultMesh);
      const int64_t meshCount = ++meshCounts[meshOfSharding.name()];
      if (meshCount > maxMeshCount) {
        maxMeshCount = meshCount;
        mostCommonMesh = meshOfSharding;
      }
    }
  }
  return mostCommonMesh;
}

// Returns the most common mesh. Returns nullopt if any of the following holds:
//  1. There is no tensor with a sharding attribute.
//  2. Tensors have different meshes (ignoring device ids)
//  3. Some tensors have maximal meshes.
std::optional<Mesh> getMesh(ArrayRef<TensorShardingAttr> inShardings,
                            ArrayRef<TensorShardingAttr> outShardings,
                            const SymbolTable& symbolTable) {
  std::optional<StringRef> meshName = getCommonMeshName(
      inShardings, outShardings, symbolTable, /*ignoreDeviceIds=*/true);
  if (!meshName.has_value()) {
    // This means none of the operands or results have a sharding attribute or
    // the sharding attributes use different meshes.
    // TODO(enver): Actually, we are moving towards supporting multiple explicit
    // reshards so operands and results are all bound by the same mesh.
    return std::nullopt;
  }
  MeshAttr meshAttr = getMeshAttr(symbolTable, *meshName);
  assert(meshAttr && "unknown mesh");
  if (meshAttr.isMaximal()) {
    return std::nullopt;
  }
  // Return the mesh with the most common device id.
  return getMostCommonMesh(inShardings, outShardings, symbolTable,
                           /*defaultMesh=*/Mesh(meshAttr, *meshName));
}

void insertAllReduceOnOpIfUnreducedToReplicated(
    Operation* op, IRRewriter& rewriter, const SymbolTable& symbolTable) {
  if (op->getResults().empty()) {
    auto operandHasUnreducedAxes = [&](OpOperand& operand) {
      TensorShardingAttr sharding = getSharding(operand.get());
      return sharding && !sharding.getUnreducedAxes().empty();
    };
    SDY_CHECK(!llvm::any_of(op->getOpOperands(), operandHasUnreducedAxes))
        << "Some operands have unreduced axes but the operation has no "
          "results. ";
    return;
  }

  TensorShardingAttr firstResultSharding = getSharding(op->getResult(0));
  if (op->getNumResults() > 1) {
    ArrayRef<AxisRefAttr> firstResultUnreducedAxes =
        getUnreducedAxes(firstResultSharding);
    for (OpResult result : op->getResults().drop_front()) {
      SDY_CHECK(firstResultUnreducedAxes == getUnreducedAxes(result))
          << "Unreduced axes mismatch between results for multi-result op.";
    }
  }

  // For each operand that has unreduced axes, insert an all-reduce if
  // any of the unreduced axes isn't unreduced in the target sharding.
  //
  // We assume all results of an op should have the same unreduced axes,
  // so we look at the first result.
  rewriter.setInsertionPoint(op);
  for (OpOperand& operand : op->getOpOperands()) {
    if (TensorShardingAttr inSharding = getSharding(operand.get())) {
      insertAllReduceIfUnreducedToReplicated(
          operand, inSharding, firstResultSharding, symbolTable, rewriter);
    }
  }
}

bool isOnFullVersion(Operation* op, const bool enableFullVersion) {
  if (enableFullVersion) {
    return true;
  }

  // The full version is disabled globally. We enable it for the following ops.
  if (op->getName().getStringRef() == "mhlo.ragged_dot") {
    return true;
  }
  // To avoid copies of the same functions with mismatching shardings on the
  // arguments onto multiple callsites.
  if (isa<NamedComputationOp>(op)) {
    return true;
  }

  // For a concatenate op, we only insert explicit reshards if any of the
  // operands has a different sharding from the result.
  //
  // We still need to insert explicit reshards if the operands and results share
  // the same sharding if concat dim is partitioned. We do not insert explicit
  // reshards in this case to avoid potential issues like b/393584711#comment3.
  if (isa<stablehlo::ConcatenateOp>(op) &&
      llvm::any_of(op->getOperands(), [&](Value operand) {
        return getSharding(operand) != getSharding(op->getResult(0));
      })) {
    return true;
  }

  return false;
}


// Inserts explicit reshards on the operands and results of `op` such that the
// sharding of `op` is compatible with its sharding rule.
//
// Refer to the documentation of `InsertExplicitReshardsPass` for more details.
//
// Assume the followings:
// - All op results have the same unreduced axes.
// - If the op has no results, none of the operands has unreduced axes.
// - Operand and result meshes are the same ignoring device id order.
// - There are no overflow axes.
//
// Returns the union of axes along all the reduction factors which may not be
// canonicalized.
//
// Guarantees to return non-empty `AxesPerFactor` if `onFullVersion` is true.
AxesPerFactor processOp(Operation* op, ShardingProjection& shardingProjection,
                        ArrayRef<TensorShardingAttr> inShardings,
                        ArrayRef<TensorShardingAttr> outShardings,
                        IRRewriter& rewriter, const SymbolTable& symbolTable,
                        OpShardingRuleAttr shardingRule, const Mesh& mesh,
                        const bool onFullVersion) {
  // Checks if factors are sharded the same way across operands and results.

  // TODO(b/446833985): Return common axes per factor also when the sharding
  // projection have overflow axes.
  if (onFullVersion) {
    AxesPerFactor commonAxesPerFactor = findCommonAxes(
        shardingProjection, shardingRule, getTensorSizes(op), mesh);

    UpdateTensorShardings updateTensorShardings(shardingRule.getNumOperands(),
                                                shardingRule.getNumResults());
    for (const auto& [index, axes] : llvm::enumerate(commonAxesPerFactor)) {
      // TODO(enver): Add unit tests to test overflow axes are cleared after
      // handling the case that some factors have overflow axes.
      updateTensorShardings |=
          shardingProjection.updateSharding(index, axes, /*overflowAxes=*/{});
    }
    insertExplicitReshards(op, inShardings, outShardings, shardingProjection,
                           updateTensorShardings, rewriter, shardingRule,
                           symbolTable, mesh);
    return commonAxesPerFactor;
  }

  TypeSwitch<Operation*>(op)
      .Case<stablehlo::DotOp>([&](stablehlo::DotOp dotOp) {
        processDot(dotOp, shardingProjection, outShardings, rewriter,
                   symbolTable, shardingRule, mesh);
      })
      .Case<stablehlo::DotGeneralOp>([&](stablehlo::DotGeneralOp dotGeneralOp) {
        processDot(dotGeneralOp, shardingProjection, outShardings, rewriter,
                   symbolTable, shardingRule, mesh);
      });
  return AxesPerFactor();
}

struct InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {
  using InsertExplicitReshardsPassBase::InsertExplicitReshardsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    SymbolTable symbolTable(funcOp->getParentOfType<ModuleOp>());

    funcOp->walk([&](Operation* op) {
      const bool onFullVersion = isOnFullVersion(op, enableFullVersion);

      if (op->hasTrait<OpTrait::IsTerminator>()) {
        if (isa<func::ReturnOp>(op)) {
          // TODO(enver): Does not need to be part of the walk on the func,
          // instead get the terminator with getBodyTerminator.
          insertExplicitReshardsOnFuncReturn(op, funcOp, rewriter, symbolTable,
                                             onFullVersion);
        }
        return;
      }

      if (auto shardableDataFlowOp =
              dyn_cast<ShardableDataFlowOpInterface>(op)) {
        // TODO(enver): Prefer resharding the owner when multiple sources are
        // sharded in the same way.
        insertExplicitReshardsOnDataFlowOp(shardableDataFlowOp, rewriter,
                                           symbolTable, onFullVersion,
                                           avoidReshardsOnNamedComputations);
        return;
      }

      insertAllReduceOnOpIfUnreducedToReplicated(op, rewriter, symbolTable);

      // NOTE: Creating a sharding rule requires data flow edges are present.
      OpShardingRuleAttr shardingRule =
          getOrCreateShardingRule(op, /*conservativePropagation=*/false,
                                  /*setShardingRuleOnOp=*/false);

      // TODO(b/434668939): Enable explicit reshards on custom sharding rules.
      if (!shardingRule || shardingRule.isCustom()) {
        // Insert explicit reshards only on operations with sharding rules,
        // since all the operations of interest got their sharding rules.
        return;
      }

      SmallVector<TensorShardingAttr> inShardings =
          getShardings(op->getOperands());
      SmallVector<TensorShardingAttr> outShardings =
          getShardings(op->getResults());

      std::optional<Mesh> mesh =
          getMesh(inShardings, outShardings, symbolTable);
      if (!mesh.has_value()) {
        return;
      }

      ShardingProjection shardingProjection = ShardingProjection::build(
          inShardings, outShardings, shardingRule, mesh->attr(),
          /*closedIfMissing=*/true);
      // Return without inserting reshards if any factor sharding has overflow
      // axes. This case is not handled yet.
      // TODO(enver): Handle the case when factor shardings have overflow axes.
      if (hasOverflowAxes(shardingProjection)) {
        return;
      }
      AxesPerFactor commonAxesPerFactor =
          processOp(op, shardingProjection, inShardings, outShardings, rewriter,
                    symbolTable, shardingRule, *mesh, onFullVersion);
      // TODO(b/440055868): Insert a reshard from unreduced to replicated axes.
      insertAllReducesForReductionFactors(op, shardingProjection,
                                          commonAxesPerFactor, shardingRule,
                                          *mesh, rewriter, onFullVersion);

      // TODO(enver): Remove sharding rules from ops.
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
