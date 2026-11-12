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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/optimize_collectives_util.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"               // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_OPTIMIZECOLLECTIVESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Pattern that optimizes collective permute preceding an all-to-all chain.
class CollectivePermuteOptimizationPattern
    : public OpRewritePattern<CollectivePermuteOp> {
 public:
  CollectivePermuteOptimizationPattern(
      MLIRContext* context, SymbolTableCollection& symbolTableCollection)
      : OpRewritePattern<CollectivePermuteOp>(context),
        symbolTableCollection(symbolTableCollection) {}

  LogicalResult matchAndRewrite(CollectivePermuteOp cpOp,
                                PatternRewriter& rewriter) const override {
    const SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(cpOp->getParentOfType<ModuleOp>());
    std::optional<int64_t> splitDim = getSplitDimension(cpOp, symbolTable);
    if (!splitDim) {
      return failure();
    }

    std::optional<AllToAllChain> chain = extractAllToAllChain(cpOp, *splitDim);
    if (!chain) {
      return failure();
    }

    if (!isChainOptimizable(*chain, *splitDim, symbolTable)) {
      return failure();
    }

    std::optional<AllToAllRewritePlan> plan =
        computeRewritePlan(*chain, *splitDim, symbolTable);
    if (!plan) {
      return failure();
    }

    return rewriteAllToAllChain(*chain, *plan, rewriter);
  }

 private:
  SymbolTableCollection& symbolTableCollection;
};

// Pattern that commutes binary reduction-preserving operations with
// sdy.all_reduce:
//   op(all_reduce(x), all_reduce(y)) -> all_reduce(op(x, y))
// when:
//   - op matches the all_reduce reduction_op (AddOp/SubtractOp for SUM,
//     MaxOp for MAX, MinOp for MIN).
//   - Both all_reduce ops have matching reduction_op, reduction_axes, and
//     out_sharding.
//   - Both all_reduce ops have a single use and matching operand shardings.
template <typename BinaryOpTy, ReductionOp kReductionOp>
class BinaryOpOfAllReducePattern : public OpRewritePattern<BinaryOpTy> {
 public:
  using OpRewritePattern<BinaryOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOpTy binaryOp,
                                PatternRewriter& rewriter) const override {
    auto lhsAllReduce = binaryOp.getLhs().template getDefiningOp<AllReduceOp>();
    auto rhsAllReduce = binaryOp.getRhs().template getDefiningOp<AllReduceOp>();
    if (!lhsAllReduce || !rhsAllReduce) {
      return failure();
    }

    if (!lhsAllReduce->hasOneUse() || !rhsAllReduce->hasOneUse()) {
      return failure();
    }

    if (lhsAllReduce.getReductionOp() != kReductionOp ||
        rhsAllReduce.getReductionOp() != kReductionOp) {
      return failure();
    }

    if (lhsAllReduce.getReductionAxes() != rhsAllReduce.getReductionAxes()) {
      return failure();
    }

    if (lhsAllReduce.getOutSharding() != rhsAllReduce.getOutSharding()) {
      return failure();
    }

    if (TensorShardingAttr outSharding = getSharding(binaryOp.getResult());
        outSharding && outSharding != lhsAllReduce.getOutSharding()) {
      return failure();
    }

    Value lhsTensor = lhsAllReduce.getTensor();
    Value rhsTensor = rhsAllReduce.getTensor();
    if (lhsTensor.getType() != rhsTensor.getType() ||
        (kReductionOp == ReductionOp::SUM &&
         getElementTypeOrSelf(lhsTensor.getType()).isInteger(1))) {
      return failure();
    }

    if (getSharding(lhsTensor) != getSharding(rhsTensor)) {
      return failure();
    }

    auto newBinaryOp = BinaryOpTy::create(
        rewriter, binaryOp.getLoc(), lhsTensor, rhsTensor);

    if (TensorShardingAttr inSharding = getSharding(lhsTensor)) {
      setSharding(newBinaryOp.getResult(), inSharding);
    }

    rewriter.replaceOpWithNewOp<AllReduceOp>(
        binaryOp, newBinaryOp.getResult(), lhsAllReduce.getReductionAxesAttr(),
        lhsAllReduce.getReductionOp(), lhsAllReduce.getOutShardingAttr());
    rewriter.eraseOp(lhsAllReduce);
    rewriter.eraseOp(rhsAllReduce);
    return success();
  }
};

struct OptimizeCollectivesPass
    : public impl::OptimizeCollectivesPassBase<OptimizeCollectivesPass> {
  using OptimizeCollectivesPassBase::OptimizeCollectivesPassBase;

 protected:
  void runOnOperation() final {
    SymbolTableCollection symbolTableCollection;
    RewritePatternSet patterns(&getContext());
    patterns.add<CollectivePermuteOptimizationPattern>(patterns.getContext(),
                                                       symbolTableCollection);
    patterns.add<
        BinaryOpOfAllReducePattern<stablehlo::AddOp, ReductionOp::SUM>,
        BinaryOpOfAllReducePattern<stablehlo::SubtractOp, ReductionOp::SUM>,
        BinaryOpOfAllReducePattern<stablehlo::MaxOp, ReductionOp::MAX>,
        BinaryOpOfAllReducePattern<stablehlo::MinOp, ReductionOp::MIN>>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
