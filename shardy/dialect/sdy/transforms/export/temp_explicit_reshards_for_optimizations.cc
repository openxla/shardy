/* Copyright 2025 The Shardy Authors.

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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_TEMPEXPLICITRESHARDSFOROPTIMIZATIONSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

bool hasOverlappingAxis(ArrayRef<AxisRefAttr> axes, AxisRefAttr axis) {
  return llvm::any_of(axes, [&](AxisRefAttr a) { return a.overlaps(axis); });
}

std::optional<ArrayRef<AxisRefAttr>> getFactorSharding(
    const TensorFactorShardings& factorShardings, int64_t factorIndex) {
  if (auto it = factorShardings.factorIndexToSharding.find(factorIndex);
      it != factorShardings.factorIndexToSharding.end()) {
    return it->second.axisRefs;
  }
  return std::nullopt;
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
void processDot(OpTy op, IRRewriter& rewriter, const SymbolTable& symbolTable) {
  SmallVector<TensorShardingAttr> inShardingAttrs =
      getShardings(op.getOperands());
  ArrayRef<TensorShardingAttr> outShardingAttrs =
      getShardings(op.getOperation());
  if (outShardingAttrs.empty()) {
    // Result doesn't have a sharding.
    return;
  }
  std::optional<StringRef> meshName =
      getCommonMeshName(inShardingAttrs, outShardingAttrs, symbolTable,
                        /*ignoreDeviceIds=*/false);
  if (!meshName.has_value()) {
    // This means none of the operands or results have a sharding attribute
    // or the sharding attributes use different meshes. Skip if so.
    return;
  }
  MeshAttr mesh = getMeshAttr(symbolTable, meshName.value());
  assert(mesh && "unknown mesh");
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(
      op.getOperation(), /*conservativePropagation=*/false,
      /*setShardingRuleOnOp=*/false);
  ShardingProjection shardingProjection =
      ShardingProjection::build(inShardingAttrs, outShardingAttrs, shardingRule,
                                mesh, /*closedIfMissing=*/true);

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
        if (hasOverlappingAxis(seenAxesAlongNonContractingDim, axis)) {
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
                  shardingRule.getFactorSizes(), meshName.value(), mesh));
  rewriter.setInsertionPointAfter(op);
  auto reshardOp = rewriter.create<ReshardOp>(op.getLoc(), op.getResult(),
                                              outShardingAttrs.front());
  rewriter.replaceAllUsesExcept(op.getResult(), reshardOp, reshardOp);
}

struct TempExplicitReshardsForOptimizationsPass
    : public impl::TempExplicitReshardsForOptimizationsPassBase<
          TempExplicitReshardsForOptimizationsPass> {
  using TempExplicitReshardsForOptimizationsPassBase::
      TempExplicitReshardsForOptimizationsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());
    SymbolTable symbolTable(funcOp->getParentOfType<ModuleOp>());
    funcOp->walk([&](Operation* op) {
      TypeSwitch<Operation*>(op)
          .Case<stablehlo::DotOp>([&](stablehlo::DotOp dotOp) {
            processDot(dotOp, rewriter, symbolTable);
          })
          .Case<stablehlo::DotGeneralOp>(
              [&](stablehlo::DotGeneralOp dotGeneralOp) {
                processDot(dotGeneralOp, rewriter, symbolTable);
              });
      if (op->getName().getStringRef() == "mhlo.ragged_dot") {
        insertExplicitReshardsOnOp(op, rewriter, symbolTable);
      }
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
