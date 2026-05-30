/* Copyright 2026 The Shardy Authors.

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

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SHARDYRESOLVEPERMUTATIONFACTORSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct ShardyResolvePermutationFactorsPass
    : public impl::ShardyResolvePermutationFactorsPassBase<
          ShardyResolvePermutationFactorsPass> {
  using ShardyResolvePermutationFactorsPassBase::
      ShardyResolvePermutationFactorsPassBase;

 protected:
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    SymbolTable symbolTable(funcOp->getParentOfType<ModuleOp>());

    // Walk the function to resolve permutation factors for each op.
    funcOp.walk([&](Operation* op) {
      // Skip terminators and any operations not in the StableHLO dialect.
      // This prevents "unknown op" warnings for Shardy collectives or return
      // ops.
      if (op->hasTrait<OpTrait::IsTerminator>() ||
          !inDialect<stablehlo::StablehloDialect>(op)) {
        return;
      }

      OpShardingRuleAttr rule = getOrCreateShardingRule(op, false, false);
      if (!rule || rule.isCustom()) {
        return;
      }

      // Identify if the op defines any permutation factors.
      auto isPermutation = [&](int64_t i) {
        return rule.getFactorType(i) == FactorType::kPermutation;
      };
      if (llvm::none_of(llvm::seq<int64_t>(0, rule.getNumFactors()),
                        isPermutation)) {
        return;
      }

      // Dispatch to HALO exchange if enabled and implemented for the op.

      // Otherwise, use a generic resolution based on explicit reshards.
      SmallVector<TensorShardingAttr> inShardings =
          getShardings(op->getOperands());
      SmallVector<TensorShardingAttr> outShardings =
          getShardings(op->getResults());
      std::optional<StringRef> meshName =
          getCommonMeshName(inShardings, outShardings, symbolTable, true);
      if (!meshName) {
        return;
      }
      MeshOp meshOp = getMeshOp(symbolTable, *meshName);
      if (!meshOp || meshOp.getMesh().isMaximal()) {
        return;
      }

      ShardingProjection projection =
          ShardingProjection::build(inShardings, outShardings, rule,
                                    meshOp.getMesh(), /*closedIfMissing=*/true);
      UpdateTensorShardings update(op->getNumOperands(), op->getNumResults());

      for (int64_t i = 0; i < rule.getNumFactors(); ++i) {
        if (rule.getFactorType(i) != FactorType::kPermutation) {
          continue;
        }
        if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(op)) {
          SDY_CHECK(inShardings[0] == outShardings[0]);
          if (isCommunicationFreeSliceDim(i, sliceOp, inShardings[0],
                                          meshOp.getMesh())) {
            continue;
          }
        }

        update |=
            projection.updateSharding(i, /*axes=*/{}, /*overflowAxes=*/{});
      }

      if (update.updateOperands.any() || update.updateResults.any()) {
        insertExplicitReshards(op, inShardings, outShardings, projection,
                               update, rewriter, rule, symbolTable, meshOp);
      }
    });
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
