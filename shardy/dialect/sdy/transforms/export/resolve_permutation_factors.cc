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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SHARDYRESOLVEPERMUTATIONFACTORSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct ResolvePermutationFactorsPattern : public RewritePattern {
  ResolvePermutationFactorsPattern(MLIRContext* ctx, bool enableHaloExchange)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx),
        enableHaloExchange(enableHaloExchange) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    OpShardingRuleAttr rule = getOrCreateShardingRule(op, false, false);
    if (!rule || rule.isCustom()) {
      return failure();
    }

    // Early exit if the rule does not define any permutation factors.
    auto isPermutation = [&](int64_t i) {
      return rule.getFactorType(i) == FactorType::kPermutation;
    };
    if (llvm::none_of(llvm::seq<int64_t>(0, rule.getNumFactors()),
                      isPermutation)) {
      return failure();
    }

    // Dispatch to HALO exchange if enabled and implemented for the op.

    // Otherwise, use a generic resolution based on explicit reshards.
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    SmallVector<TensorShardingAttr> inShardings =
        getShardings(op->getOperands());
    SmallVector<TensorShardingAttr> outShardings =
        getShardings(op->getResults());
    std::optional<StringRef> meshName =
        getCommonMeshName(inShardings, outShardings, symbolTable, true);
    if (!meshName) {
      return failure();
    }
    MeshOp meshOp = getMeshOp(symbolTable, *meshName);
    if (!meshOp || meshOp.getMesh().isMaximal()) {
      return failure();
    }

    ShardingProjection projection =
        ShardingProjection::build(inShardings, outShardings, rule,
                                  meshOp.getMesh(), /*closedIfMissing=*/true);
    UpdateTensorShardings update(op->getNumOperands(), op->getNumResults());

    for (int64_t i = 0; i < rule.getNumFactors(); ++i) {
      if (rule.getFactorType(i) == FactorType::kPermutation) {
        if (!enableHaloExchange) {
          update |=
              projection.updateSharding(i, /*axes=*/{}, /*overflowAxes=*/{});
        }
      }
    }
    if (update.updateOperands.any() || update.updateResults.any()) {
      IRRewriter sdyRewriter(rewriter);
      insertExplicitReshards(op, inShardings, outShardings, projection, update,
                             sdyRewriter, rule, symbolTable, meshOp);
      return success();
    }

    return failure();
  }

  bool enableHaloExchange;
};

struct ShardyResolvePermutationFactorsPass
    : public impl::ShardyResolvePermutationFactorsPassBase<
          ShardyResolvePermutationFactorsPass> {
  using ShardyResolvePermutationFactorsPassBase::
      ShardyResolvePermutationFactorsPassBase;

 protected:
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ResolvePermutationFactorsPattern>(&getContext(),
                                                   enableHaloExchange);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace
}  // namespace sdy
}  // namespace mlir
