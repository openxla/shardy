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

      // TODO(enver): Build a projection where, for each factor, factor
      // shardings are the same across all operands and results;

      // TODO(enver): Insert the explicit reshard ops.
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
