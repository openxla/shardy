/* Copyright 2025 The MPMD Authors.

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

#include <utility>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/simplify_region_op_base.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_SIMPLIFYNAMEDCOMPUTATIONSPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

using ::mlir::stablehlo::OptimizationBarrierOp;

class SimplifyNamedComputationPattern
    : public SimplifyRegionOpPatternBase<NamedComputationOp> {
  using SimplifyRegionOpPatternBase::SimplifyRegionOpPatternBase;

  NamedComputationOp createNewOp(NamedComputationOp op,
                                 PatternRewriter& rewriter,
                                 TypeRange result_types, ValueRange operands,
                                 BitVector erased_results) const override {
    return NamedComputationOp::create(rewriter, op.getLoc(), result_types,
                                      operands, op.getOriginAttr());
  }
};

// This pattern replaces the pattern within a NamedComputationOp
// `arg -> OptBarrier -> return` with the pattern `arg -> return`, allowing
// further simplification of the NamedComputation.
//
// This may happen when jax.remat is used and unrelated args are thus captured
// in the NamedComputation.
//
// It is generally safe to apply, since between programs, the OptBarrier is a
// no-op, although we may end up merging with other ops which may need the
// OptBarrier. But we assume that only values that are actually used are what we
// need the OptBarrier on.
class SimplifyOptimizationBarrier
    : public OpRewritePattern<OptimizationBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(OptimizationBarrierOp op,
                                PatternRewriter& rewriter) const override {
    auto named_comp_parent = dyn_cast<NamedComputationOp>(op->getParentOp());
    if (!named_comp_parent) {
      return rewriter.notifyMatchFailure(
          op, [&](Diagnostic& diag) { diag << "Not in a named computation"; });
    }

    llvm::BitVector kept_operands(op->getNumOperands());
    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) {
        OpResult res = op->getResult(idx);
        // We could extend this to handle multiple uses, but it's not clear
        // that's needed.
        if (res.hasOneUse() && isa<ReturnOp>(*res.getUsers().begin())) {
          rewriter.replaceAllUsesWith(res, operand);
          continue;
        }
      }
      kept_operands.set(idx);
    }

    if (kept_operands.count() == op->getNumOperands()) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "All operands are kept, nothing to do.";
      });
    }

    llvm::SmallVector<Value> new_operands;
    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      if (kept_operands.test(idx)) {
        new_operands.push_back(operand);
      }
    }
    // Replace opt_barrier op with a new one with the kept operands.
    auto opt_barrier =
        OptimizationBarrierOp::create(rewriter, op.getLoc(), new_operands);

    int opt_barrier_idx = 0;
    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      if (!kept_operands.test(idx)) {
        continue;
      }
      rewriter.replaceAllUsesWith(op->getResult(idx),
                                  opt_barrier->getResult(opt_barrier_idx));
      opt_barrier_idx++;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class SimplifyNamedComputationsPass
    : public impl::SimplifyNamedComputationsPassBase<
          SimplifyNamedComputationsPass> {
  using SimplifyNamedComputationsPassBase::SimplifyNamedComputationsPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    {
      RewritePatternSet patternsInternal(context);
      patternsInternal.add<SimplifyOptimizationBarrier>(context);
      opt_barrier_patterns = std::move(patternsInternal);
    }

    {
      RewritePatternSet patternsInternal(context);
      patternsInternal.add<SimplifyNamedComputationPattern>(context);
      named_computation_patterns = std::move(patternsInternal);
    }

    return success();
  }

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());

    // NOTE: We may possibly want to do this in JAXPR. Apply these passes first
    // to avoid repeated simplification.
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled)
        .enableFolding(false)
        .enableConstantCSE(false);
    if (failed(applyPatternsGreedily(func_op, opt_barrier_patterns, config))) {
      return signalPassFailure();
    }

    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
    if (failed(applyPatternsGreedily(func_op, named_computation_patterns,
                                     config))) {
      return signalPassFailure();
    }
  }

  FrozenRewritePatternSet opt_barrier_patterns;
  FrozenRewritePatternSet named_computation_patterns;
};

}  // namespace

}  // namespace mlir::mpmd
