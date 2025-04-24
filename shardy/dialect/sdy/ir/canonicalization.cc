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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

#include "shardy/dialect/sdy/ir/canonicalization.cc.inc"

// Pattern to remove unused block arguments and their corresponding operands
// from  a `ManualComputationOp`.
class ManualComputationUnusedInputsPattern
    : public OpRewritePattern<ManualComputationOp> {
 public:
  using OpRewritePattern<ManualComputationOp>::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(ManualComputationOp manualComputationOp,
                                PatternRewriter& rewriter) const override {
    BitVector unusedArgs(manualComputationOp.getNumOperands());
    for (BlockArgument arg : manualComputationOp.getRegion().getArguments()) {
      if (arg.use_empty()) {
        unusedArgs.set(arg.getArgNumber());
      }
    }
    if (unusedArgs.none()) {
      return failure();
    }

    manualComputationOp->eraseOperands(unusedArgs);
    manualComputationOp.getRegion().front().eraseArguments(unusedArgs);

    SmallVector<TensorShardingAttr> inShardings;
    inShardings.reserve(manualComputationOp.getNumOperands());
    for (int64_t index : unusedArgs.flip().set_bits()) {
      inShardings.push_back(manualComputationOp.getInSharding(index));
    }
    manualComputationOp.setInShardings(inShardings);

    return success();
  }
};

// Pattern to inline a ManualComputationOp when the product of all manual axes
// is 1.
class RedundantManualComputationPattern
    : public OpRewritePattern<ManualComputationOp> {
 public:
  using OpRewritePattern<ManualComputationOp>::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(ManualComputationOp manualComputationOp,
                                PatternRewriter& rewriter) const override {
    ArrayRef<TensorShardingAttr> inShardings =
        manualComputationOp.getInShardings().getShardings();
    ArrayRef<TensorShardingAttr> outShardings =
        manualComputationOp.getOutShardings().getShardings();

    int64_t manualAxesProduct = 1;
    if (!inShardings.empty() || !outShardings.empty()) {
      MeshAttr mesh =
          getCommonMesh(inShardings, outShardings, manualComputationOp);
      assert(mesh && "expected inputs and outputs to have a common mesh");
      for (StringAttr manualAxis : manualComputationOp.getManualAxes()) {
        manualAxesProduct *= mesh.getAxisSize(manualAxis);
      }
    }

    if (manualAxesProduct != 1) {
      return rewriter.notifyMatchFailure(
          manualComputationOp, [](Diagnostic& diag) {
            diag << "product of manual axis sizes is not 1";
          });
    }

    mlir::InlinerInterface inliner(manualComputationOp.getContext());
    mlir::InlinerConfig config;
    if (inlineRegion(
            inliner, config.getCloneCallback(),
            &manualComputationOp.getRegion(), manualComputationOp->getBlock(),
            manualComputationOp->getIterator(),
            manualComputationOp.getOperands(), manualComputationOp.getResults())
            .failed()) {
      manualComputationOp.emitOpError(
          "failed to inline redundant ManualComputationOp.");
      return failure();
    }
    rewriter.eraseOp(manualComputationOp);
    return success();
  }
};

}  // namespace

void ManualComputationOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<ManualComputationUnusedInputsPattern,
              RedundantManualComputationPattern>(context);
}

void ReshardOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<ReshardOfReshardPattern>(context);
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<AllGatherNoopPattern>(context);
}

void AllSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<AllSliceOfAllGatherPattern>(context);
  results.add<AllSliceNoopPattern>(context);
}

void AllReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<AllReduceNoopPattern>(context);
}

void AllToAllOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<AllToAllNoopPattern>(context);
}

void CollectivePermuteOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<CollectivePermuteNoopPattern>(context);
}

}  // namespace sdy
}  // namespace mlir
