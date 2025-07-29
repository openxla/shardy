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

#include <memory>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_CONVERTSDYCONSTANTSPASS
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h.inc"

namespace {

// Converts sdy::ConstantOp to stablehlo::ConstantOp.
class ConstantPattern : public OpConversionPattern<sdy::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      sdy::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // We use the generic op builder so that unregistered attributes will be
    // added to the new op.
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op->getResultTypes(), adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

class ConvertSdyConstantsPass
    : public impl::ConvertSdyConstantsPassBase<ConvertSdyConstantsPass> {
  using ConvertSdyConstantsPassBase::ConvertSdyConstantsPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<sdy::ConstantOp>();
    target->addLegalOp<stablehlo::ConstantOp>();

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<ConstantPattern>(context);
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
}  // namespace mlir::mpmd
