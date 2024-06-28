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
#include <memory>  // IWYU pragma: keep
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SINKDATAFLOWEDGESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// This pattern matches on a specific `DataFlowEdgeOp`, but will also sink any
// other `DataFlowEdgeOp` whose input is defined by the same op. This way we can
// build the `TensorShardingPerValueAttr` for the defining op once.
class SinkDataFlowEdgesPattern : public OpRewritePattern<DataFlowEdgeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(DataFlowEdgeOp dataFlowEdgeOp,
                                PatternRewriter& rewriter) const override {
    Operation* defOp = dataFlowEdgeOp.getInput().getDefiningOp();
    if (!defOp) {
      // `dataFlowEdgeOp` takes a block argument, we ignore the sharding of
      // `dataFlowEdgeOp` since a block argument can't have a sharding attached.
      // TODO(tomnatan): we might need to revisit this for future use cases.
      rewriter.replaceOp(dataFlowEdgeOp, dataFlowEdgeOp.getInput());
      return success();
    }

    SmallVector<TensorShardingAttr> shardings(defOp->getNumResults());

    // For each result of `defOp` that is used by a `DataFlowEdgeOp`:
    // - If the `DataFlowEdgeOp` has a sharding, add it to `shardings`.
    // - Replace the `DataFlowEdgeOp` with its input.
    //
    // In addition, stores the mesh name of first encountered sharding, as we
    // need a mesh name to replace missing shardings with fully replicated
    // shardings. Note that it's ok to pick an arbitrary mesh if there are
    // multiple, as we are creating fully replicated shardings.
    StringRef meshName;
    for (auto [index, result] : llvm::enumerate(defOp->getResults())) {
      // We can assume a `DataFlowEdgeOp` will be the only user of its input.
      DataFlowEdgeOp dataFlowEdgeOp =
          DataFlowEdgeOp::getDataFlowEdgeUser(result);
      if (!dataFlowEdgeOp) {
        continue;
      }
      if (TensorShardingAttr sharding = dataFlowEdgeOp.getShardingAttr()) {
        shardings[index] = sharding;
        if (meshName.empty()) {
          meshName = sharding.getMeshName();
        }
      }
      rewriter.replaceOp(dataFlowEdgeOp, dataFlowEdgeOp.getInput());
    }

    if (!meshName.empty()) {
      // There is at least one `DataFlowEdgeOp` with a sharding.
      // Replace all empty shardings with fully open shardings.
      for (auto [sharding, result] :
           llvm::zip(shardings, defOp->getResults())) {
        if (!sharding) {
          sharding = getOrCreateSharding(result, meshName);
        }
      }
      defOp->setAttr(kShardingAttr, TensorShardingPerValueAttr::get(
                                        defOp->getContext(), shardings));
    }

    return success();
  }
};

struct SinkDataFlowEdgesPass
    : public impl::SinkDataFlowEdgesPassBase<SinkDataFlowEdgesPass> {
  using SinkDataFlowEdgesPassBase::SinkDataFlowEdgesPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    config.useTopDownTraversal = true;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    config.maxIterations = 2;

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<SinkDataFlowEdgesPattern>(context);
    patterns = std::move(patternsInternal);

    return success();
  }

  void runOnOperation() final {
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), patterns, config))) {
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
