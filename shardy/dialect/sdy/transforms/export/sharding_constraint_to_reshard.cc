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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SHARDINGCONSTRAINTTORESHARDPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

class ShardingConstraintPattern
    : public OpConversionPattern<ShardingConstraintOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      ShardingConstraintOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ReshardOp>(op, adaptor.getInput())
        .setShardingAttr(adaptor.getSharding());
    return success();
  }
};

struct ShardingConstraintToReshardPass
    : public impl::ShardingConstraintToReshardPassBase<
          ShardingConstraintToReshardPass> {
  using ShardingConstraintToReshardPassBase::
      ShardingConstraintToReshardPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<ShardingConstraintOp>();
    target->addLegalOp<ReshardOp>();

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<ShardingConstraintPattern>(context);
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
