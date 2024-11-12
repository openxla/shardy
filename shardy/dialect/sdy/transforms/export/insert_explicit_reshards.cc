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
#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"    // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"      // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Returns true iff any tensor factor sharding has non-empty overflow axes.
bool hasOverflowAxes(const ShardingProjection& projection) {
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [_, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (!factorSharding.overflowAxes.empty()) {
        return true;
      }
    }
  }
  return false;
}

// Checks if factor sharding is compatible, that is, it satisfies:
// 1. Factors are sharded the same way across operands and results.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
bool hasCompatibleFactorShardings(const ShardingProjection& projection) {
  FactorIndexToSharding factorIndexToCommonSharding;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    // Detects conflicts within the same factor.
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      auto commonFactorShardingIt =
          factorIndexToCommonSharding.find(factorIndex);
      if (commonFactorShardingIt == factorIndexToCommonSharding.end()) {
        factorIndexToCommonSharding[factorIndex] = factorSharding;
        continue;
      }
      if (factorSharding.axisRefs != commonFactorShardingIt->second.axisRefs) {
        return false;
      }
    }
  }

  // TODO(enver): Detect conflicts across different factors.
  return true;
}

// Insert explicit reshards for operands and results that change by
// the given `projection` for a given `op`. The reshards are inserted only to
// make the given operation compatible.
//
// For example,
//
// ```mlir
//   %arg0: tensor<8x32xf32> { sdy.sharding = @mesh, [{}, {"y"}]>}
//   %arg1: tensor<32x16xf32> { sdy.sharding = <@mesh, [{"y"}, {"x"}]>}
//   %0 = stablehlo.dot %arg0, %arg1 { sdy.sharding = <@mesh, [{"x"}, {}]>,
//     sdy.sharding_rule = <([i, k], [k, j])->([i, j])> }
//   %1 = stablehlo.negate %0 {sdy.sharding = <@mesh, [{"x"}, {}]>
//   return %1
// ```
//
// after a call on the stablehlo.dot operation, by the projection, i: {}, j: {},
// k: {"y"}, the module becomes:
//
// ```mlir
//   %arg0: tensor<8x32xf32> { sdy.sharding = @mesh, [{}, {"y"}]>}
//   %arg1: tensor<32x16xf32> { sdy.sharding = <@mesh, [{"y"}, {"x"}]>}
//   %0 = stablehlo.reshard %arg1 {sdy.sharding = <@mesh, [{"y"}, {}]>}
//   %1 = stablehlo.dot %arg0, %0 { sdy.sharding = <@mesh, [{}, {}]>,
//     sdy.sharding_rule = <([i, k], [k, j])->([i, j])> }
//   %2 = stablehlo.reshard %1 {sdy.sharding = <@mesh, [{"x"}, {}]>}
//   %3 = stablehlo.negate %2 {sdy.sharding = <@mesh, [{"x"}, {}]>
//   return %3
// ```
//
// In the above example, note that the operand and result shardings for
// stablehlo.negate op remained unchanged.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
void insertExplicitReshards(Operation* op, const ShardingProjection& projection,
                            UpdateTensorShardings updateTensorShardings,
                            IRRewriter& rewriter,
                            OpShardingRuleAttr shardingRule, StringRef meshName,
                            MeshAttr mesh) {
  rewriter.setInsertionPoint(op);
  for (int operandIndex : updateTensorShardings.updateOperands.set_bits()) {
    auto operand = op->getOperand(operandIndex);
    auto newTensorSharding =
        projection.getOperand(operandIndex)
            .createTensorShardingAttr(
                mesh.getContext(), shardingRule.getOperandMapping(operandIndex),
                shardingRule.getFactorSizes(), meshName, mesh);
    auto reshardOp = rewriter.create<ReshardOp>(operand.getLoc(), operand,
                                                newTensorSharding);
    op->setOperand(operandIndex, reshardOp);
  }

  rewriter.setInsertionPointAfter(op);
  for (int resultIndex : toSetBitsVector(updateTensorShardings.updateResults)) {
    auto result = op->getResult(resultIndex);
    auto newTensorSharding =
        projection.getResult(resultIndex)
            .createTensorShardingAttr(
                mesh.getContext(), shardingRule.getResultMapping(resultIndex),
                shardingRule.getFactorSizes(), meshName, mesh);
    auto reshardOp = rewriter.create<ReshardOp>(result.getLoc(), result,
                                                getSharding(result));
    rewriter.replaceAllUsesExcept(result, reshardOp, reshardOp);
    setSharding(result, newTensorSharding);
  }
}

// Broadly the algorithm is, at each iteration, to pick a {factor,axis} pair
// with the largest count from a list that is initialized with all the
// pairs with non-zero count, assign the picked axis to the picked factor, and
// delete all the pairs from the list that is either with the picked factor, or
// with an axis that overlaps with the picked axis. Continue iterating until the
// list is empty.
AxesPerFactor findCommonAxesUsingMajorityVoteHeuristic(
    const ShardingProjection& projection, int64_t numFactors) {
  AxesPerFactor factorAxisRefs(numFactors);
  // TODO(enver): The algorithm iterates over all {axes, factor} pairs at each
  // pick, hence the complexity is roughly O(fp) where f is the number of
  // factors, and p is the number of pairs. Optimize to lower the complextiy.
  // Consider using a map from AxisRefs to factors.
  SmallVector<DenseMap<AxisRefAttr, int64_t>> factorAxesCounts(numFactors);
  int64_t maxCount = 0;
  int64_t bestFactorIndex;
  AxisRefAttr bestAxisRef;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (factorSharding.axisRefs.empty()) {
        continue;
      }
      AxisRefAttr axisRef = factorSharding.axisRefs[0];
      int64_t axesCount = ++factorAxesCounts[factorIndex][axisRef];
      if (axesCount > maxCount) {
        maxCount = axesCount;
        bestFactorIndex = factorIndex;
        bestAxisRef = axisRef;
      }
    }
  }

  // TODO(enver): Assign an axis to a factor immediately if the count is more
  // than floor(n/2) where n is the number of tensors.
  BitVector unseenFactors(numFactors, true);
  // TODO(enver): Optimize to mark unseen only the factors with an axis.
  while (maxCount > 0) {
    factorAxisRefs[bestFactorIndex].push_back(bestAxisRef);
    unseenFactors.reset(bestFactorIndex);
    // TODO(enver): Tie-breaking currently depends on the order of iteration.
    // Consider some heuristic for breaking ties.
    // Invalidate axes that overlaps with the picked one across all unseen
    // factors. During the iteration, also find the new best.
    maxCount = 0;
    int64_t nextBestFactorIndex;
    AxisRefAttr nextBestAxisRef;
    for (int factorIndex : unseenFactors.set_bits()) {
      auto& axesCounts = factorAxesCounts[factorIndex];
      for (const auto& [axisRef, count] : axesCounts) {
        // TODO(enver): Relax the overlap check. We need to erase in case of an
        // overlap only if the factor indices appear together in any of the
        // operands or results.
        if (bestAxisRef.overlaps(axisRef)) {
          // TODO(enver): Optimize to flip unseen if all the axes of the factor
          // have zero count.
          // Clear the count of overlapping axis, effectively erasing.
          axesCounts[axisRef] = 0;
          continue;
        }
        if (count > maxCount) {
          maxCount = count;
          nextBestFactorIndex = factorIndex;
          nextBestAxisRef = axisRef;
        }
      }
    }
    bestFactorIndex = nextBestFactorIndex;
    bestAxisRef = nextBestAxisRef;
  }
  return factorAxisRefs;
}

bool hasFactorShardingWithMultipleAxesOrSubAxes(
    const ShardingProjection& projection) {
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [_, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (factorSharding.axisRefs.size() > 1) {
        return true;
      }
    }
  }
  return false;
}

AxesPerFactor findCommonAxes(const ShardingProjection& projection,
                             int64_t numFactors) {
  // TODO(enver): Use majority vote heuristic also for cases where some
  // factors may have multple axes or subaxes.
  if (hasFactorShardingWithMultipleAxesOrSubAxes(projection)) {
    return projection.getGreatestCommonPrefixAxes(numFactors);
  }

  return findCommonAxesUsingMajorityVoteHeuristic(projection, numFactors);
}

struct InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {
  using InsertExplicitReshardsPassBase::InsertExplicitReshardsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    SymbolTable symbolTable(funcOp->getParentOfType<ModuleOp>());
    // TODO(enver): Handle data flow ops.
    funcOp.walk([&](Operation* op) {
      // TODO(enver): Check if data flow ops, data flow edge op, manual
      // computation op require extra check before creating sharding rule.

      if (isa<func::ReturnOp>(op)) {
        rewriter.setInsertionPoint(op);
        for (const auto& [index, opOperand] :
             llvm::enumerate(op->getOpOperands())) {
          Value operand = opOperand.get();
          TensorShardingAttr funcResultSharding =
              getFuncResultSharding(funcOp, index);
          TensorShardingAttr operandSharding = getSharding(operand);
          if (isFullyReplicated(operandSharding) &&
              isFullyReplicated(funcResultSharding)) {
            continue;
          }
          if (funcResultSharding != operandSharding) {
            // TODO(enver): Close all shardings and drop replicated axes before
            // this pass on the export pipeline.
            auto reshardOp = rewriter.create<ReshardOp>(
                operand.getLoc(), operand,
                funcResultSharding
                    ? funcResultSharding
                    : TensorShardingAttr::getFullyClosedLike(operandSharding));
            opOperand.set(reshardOp);
          }
        }
        return;
      }

      // NOTE: Creating a sharding rule requires data flow edges are present.
      OpShardingRuleAttr shardingRule =
          getOrCreateShardingRule(op, /*conservativePropagation=*/false,
                                  /*setShardingRuleOnOp=*/false);
      if (!shardingRule) {
        // Insert explicit reshards only on operations with sharding rules,
        // since all the operations of interest got their sharding rules.
        return;
      }
      std::optional<StringRef> meshName =
          getCommonMeshName(getShardings(op->getOperands()),
                            getShardings(op->getResults()), symbolTable);
      if (!meshName.has_value()) {
        // This means none of the operands or results have a sharding attribute
        // or the sharding attributes use different meshes. Skip if so.
        // TODO(enver): Actually, we are moving towards supporting multiple
        // explicit reshards so operands and results are all bound by the same
        // mesh.
        return;
      }

      MeshAttr mesh = getMeshAttr(op, meshName.value());
      assert(mesh && "unknown mesh");
      ShardingProjection shardingProjection =
          ShardingProjection::build(op, shardingRule, mesh);

      // Return without inserting reshards if any factor sharding has overflow
      // axes. This case is not handled yet.
      // TODO(enver): Handle the case when factor shardings have overflow axes.
      if (hasOverflowAxes(shardingProjection)) {
        return;
      }

      // Checks if factors are sharded the same way across operands and results.
      if (hasCompatibleFactorShardings(shardingProjection)) {
        return;
      }

      UpdateTensorShardings updateTensorShardings(shardingRule.getNumOperands(),
                                                  shardingRule.getNumResults());
      for (const auto& [index, factorAxes] : llvm::enumerate(findCommonAxes(
               shardingProjection, shardingRule.getNumFactors()))) {
        // TODO(enver): Add unit tests to test overflow axes are cleared after
        // handling the case that some factors have overflow axes.
        updateTensorShardings |= shardingProjection.updateSharding(
            index, factorAxes, /*overflowAxes=*/{});
      }

      insertExplicitReshards(op, shardingProjection, updateTensorShardings,
                             rewriter, shardingRule, *meshName, mesh);

      // TODO(enver): Remove sharding rules from ops.
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
