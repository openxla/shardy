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
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "stablehlo/dialect/StablehloOps.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Returns true iff any tensor factor sharding has non-empty overflow axes.
bool containsOverflowAxes(const ShardingProjection& projection) {
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (!factorSharding.overflowAxes.empty()) {
        return false;
      }
    }
  }
  return true;
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

// Insert explicit reshards for operands and result tensors that change by
// the given `projection` for a given `op`. The reshards are inserted only to
// make the given operation compatible.
//
// For example,
//
// func.func @foo(
//   %arg0: tensor<8x32xf32> {
//     sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
//   %arg1: tensor<32x16xf32> {
//     sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})
//   -> tensor<8x16xf32> {
//
//   %0 = stablehlo.negate %arg1 {sdy.sharding =
//     #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x16xf32>
//   %1 = stablehlo.dot %arg0, %0 {
//     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>,
//     sdy.sharding_rule =
//       #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} :
//     (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
//   %2 = stablehlo.negate %1 {sdy.sharding =
//     #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
//   return %2 : tensor<8x16xf32>
// }
//
// after a call on the stablehlo.dot operation, by the projection, i: {}, j: {},
// k: {"y"}, the module becomes:
//
// func.func @foo(
//   %arg0: tensor<8x32xf32> {
//     sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
//   %arg1: tensor<32x16xf32> {
//     sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})
//   -> tensor<8x16xf32> {
//
//   %0 = stablehlo.negate %arg1 {sdy.sharding =
//     #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x16xf32>
//   %1 = sdy.reshard %0 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
//   %2 = stablehlo.dot %arg0, %1 {
//     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>,
//     sdy.sharding_rule =
//       #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} :
//     (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
//   %3 = sdy.reshard %2 <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
//   %4 = stablehlo.negate %3 {sdy.sharding =
//     #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
//   return %4 : tensor<8x16xf32>
// }
//
// In the above example, note that the operand and result shardings for
// stablehlo.negate ops remained unchanged.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
void insertExplicitReshards(Operation* op, const ShardingProjection& projection,
                            IRRewriter& rewriter,
                            OpShardingRuleAttr shardingRule, StringRef meshName,
                            MeshAttr mesh) {
  for (int index = op->getNumOperands() - 1; index >= 0; index--) {
    auto value = op->getOperand(index);
    rewriter.setInsertionPointAfterValue(value);
    auto newTensorSharding =
        projection.getOperand(index).createTensorShardingAttr(
            mesh.getContext(), shardingRule.getOperandMapping(index),
            shardingRule.getFactorSizes(), meshName, mesh);
    if (newTensorSharding == getSharding(value)) {
      continue;
    }
    auto reshardOp =
        rewriter.create<ReshardOp>(value.getLoc(), value, newTensorSharding);
    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(index, reshardOp); });
  }

  for (const auto& [value, tensorFactorShardings, tensorMapping] :
       llvm::zip_equal(op->getResults(), projection.getResults(),
                       shardingRule.getResultMappings())) {
    auto newTensorSharding = tensorFactorShardings.createTensorShardingAttr(
        mesh.getContext(), tensorMapping, shardingRule.getFactorSizes(),
        meshName, mesh);
    if (newTensorSharding == getSharding(value)) {
      continue;
    }
    rewriter.setInsertionPointAfterValue(value);
    auto reshardOp =
        rewriter.create<ReshardOp>(value.getLoc(), value, getSharding(value));
    rewriter.replaceAllUsesExcept(value, reshardOp, reshardOp);
    setSharding(value, newTensorSharding);
  }
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
      // TODO(enver): Handle the case when the operation does not have sharding
      // rule, perhaps use getOrCreateShardingRule utility.
      OpShardingRuleAttr shardingRule =
          op->getAttrOfType<OpShardingRuleAttr>(kShardingRuleAttr);
      if (!shardingRule) {
        // Insert explicit reshards only on operations with sharding rules,
        // since all the operations of interest got their sharding rules
        // populated at a previous populate-op-sharding-rules pass.
        return;
      }
      std::optional<StringRef> meshName =
          getCommonMeshName(getShardings(op->getOperands()),
                            getShardings(op->getResults()), symbolTable);
      if (!meshName.has_value()) {
        // This means none of the operands or results have a sharding attribute
        // or the sharding attributes use different meshes. Skip if so.
        // TODO(enver): Actually, we are moving towards supporting multiple
        // meshes during propagation. We should handle this by inserting
        // explicit reshards so operands and results are all bound by the same
        // mesh.
        return;
      }
      MeshAttr mesh = getMeshAttr(op, meshName.value());
      assert(mesh && "unknown mesh");
      ShardingProjection shardingProjection =
          ShardingProjection::build(op, shardingRule, mesh);

      // Return without inserting reshards if any factor shardings have overflow
      // axes. This case is not handled yet.
      // TODO(enver): Handle the case when factor shardings have overflow axes.
      if (containsOverflowAxes(shardingProjection)) {
        return;
      }

      // Checks if factors are sharded the same way across operands and results.
      if (hasCompatibleFactorShardings(shardingProjection)) {
        return;
      }

      // TODO(enver): Build a projection where, for each factor, factor
      // shardings are the same across all operands and results;

      insertExplicitReshards(op, shardingProjection, rewriter, shardingRule,
                             *meshName, mesh);
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
