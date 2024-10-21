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

// Checks if factor sharding is compatible, that is, it satisfies:
// 1. Factors are sharded the same way across operands and results.
bool hasCompatibleFactorSharding(const ShardingProjection& projection) {
  FactorIndexToSharding factorIndexToCommonSharding;
  for (const TensorFactorShardingMap& tensorFactorSharding :
       llvm::concat<const TensorFactorShardingMap>(projection.getOperands(),
                                                   projection.getResults())) {
    // Detects conflicts within the same factor.
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      // TODO(enver): Handle the case when some factor shardings have overflow
      // axes.
      if (!factorSharding.factor.overflowAxes.empty()) {
        return false;
      }
      auto commonFactorShardingIt =
          factorIndexToCommonSharding.find(factorIndex);
      if (commonFactorShardingIt == factorIndexToCommonSharding.end()) {
        factorIndexToCommonSharding[factorIndex] = factorSharding;
        continue;
      }
      if (factorSharding.factor.axisRefs !=
          commonFactorShardingIt->second.factor.axisRefs) {
        return false;
      }
    }
  }

  // TODO(enver): Detect conflicts across different factors.
  return true;
}

struct InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {
  using InsertExplicitReshardsPassBase::InsertExplicitReshardsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
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
      // Checks if factors are sharded the same way across operands and results.
      if (hasCompatibleFactorSharding(shardingProjection)) {
        return;
      }

      // TODO(enver): Insert the explicit reshard ops.
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
