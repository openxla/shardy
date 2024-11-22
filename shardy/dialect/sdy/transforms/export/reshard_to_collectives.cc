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
#include <cstddef>
#include <memory>  // IWYU pragma: keep
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_RESHARDTOCOLLECTIVESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

class ReshardPattern : public OpConversionPattern<ReshardOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  // For the moment we only consider all_gather.
  // If the reshard can't be expressed exclusively as an all_gather, we fail.
  // TODO(b/380226848): Add support for other collectives.
  LogicalResult matchAndRewrite(
      ReshardOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    TensorShardingAttr inputSharding = getSharding(adaptor.getInput());
    TensorShardingAttr outputSharding = adaptor.getSharding();
    // Here it's safe to assume that shardings' meshes have a name.
    if (inputSharding.getRank() != outputSharding.getRank() ||
        inputSharding.getMeshName() != outputSharding.getMeshName()) {
      return rewriter.notifyMatchFailure(
          op, [](Diagnostic& diag) { diag << "Uncompatible shardings"; });
    }
    auto gatheringAxesPerDim =
        SmallVector<AxisRefListAttr>(inputSharding.getRank());
    size_t dimIdx = 0;
    for (auto [inDimSharding, outDimSharding] :
         llvm::zip_equal(inputSharding.getDimShardings(),
                         outputSharding.getDimShardings())) {
      // We look for the greatest common prefix between the input and output
      // shardings. The gathering axes should be the output sharding without
      // this prefix.
      ArrayRef<AxisRefAttr> inShardingDimAxes = inDimSharding.getAxes();
      ArrayRef<AxisRefAttr> outShardingDimAxes = outDimSharding.getAxes();
      SmallVector<AxisRefAttr> greatestCommonPrefix =
          getGreatestCommonPrefix(inShardingDimAxes, outShardingDimAxes);
      if (greatestCommonPrefix != outShardingDimAxes) {
        return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
          diag << "Couldn't match reshard as all_gather";
        });
      }
      // TODO (b/379838852): Add support for gathering a sub axis out of a
      // bigger or full sub axis or full axis.
      ArrayRef<AxisRefAttr> gatheringAxes =
          inShardingDimAxes.drop_front(greatestCommonPrefix.size());
      gatheringAxesPerDim[dimIdx] =
          AxisRefListAttr::get(rewriter.getContext(), gatheringAxes);
      dimIdx++;
    }
    rewriter.replaceOpWithNewOp<AllGatherOp>(
        op, adaptor.getInput(),
        ListOfAxisRefListsAttr::get(rewriter.getContext(), gatheringAxesPerDim),
        adaptor.getSharding());
    return success();
  }
};

struct ReshardToCollectivesPass
    : public impl::ReshardToCollectivesPassBase<ReshardToCollectivesPass> {
  using ReshardToCollectivesPassBase::ReshardToCollectivesPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<ReshardOp>();
    target->addLegalOp<AllGatherOp>();

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<ReshardPattern>(context);
    patterns = std::move(patternsInternal);

    return success();
  }

  void runOnOperation() final {
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      signalPassFailure();
    }
  }

 private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
