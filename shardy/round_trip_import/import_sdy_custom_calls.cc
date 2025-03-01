/* Copyright 2025 The Shardy Authors.

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

#include "shardy/round_trip_import/import_sdy_custom_calls.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

using ::mlir::stablehlo::CustomCallOp;
using ::mlir::stablehlo::CustomCallOpAdaptor;

LogicalResult rewriteShardingCustomCall(CustomCallOp op,
                                        CustomCallOpAdaptor adaptor,
                                        ConversionPatternRewriter& rewriter) {
  if (op->getNumResults() != 1) {
    op.emitError() << "expected CustomCallOp with exactly one result";
    return failure();
  }
  TensorShardingAttr sharding = getSharding(op->getResult(0));
  if (!sharding) {
    op.emitError() << "expected CustomCallOp with a sharding attribute";
    return failure();
  }

  rewriter.replaceOpWithNewOp<ShardingConstraintOp>(
      op, adaptor.getInputs().front(), sharding);

  return success();
}

LogicalResult rewriteShardingGroupCustomCall(
    CustomCallOp op, CustomCallOpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) {
  assert(op.getNumOperands() == 1);
  assert(op.getNumResults() <= 1);
  std::optional<IntegerAttr> shardingGroupId =
      tryGetFrontendAttr<IntegerAttr>(op, kShardingGroupIdAttr);
  if (!shardingGroupId.has_value()) {
    return op.emitError() << "expected CustomCallOp with a sharding group id.";
  }
  if (!op.use_empty()) {
    return op.emitError()
           << "xla.sdy.ShardingGroup CustomCallOp should have no uses.";
  }

  rewriter.replaceOpWithNewOp<ShardingGroupOp>(op, adaptor.getInputs().front(),
                                               shardingGroupId->getInt());

  return success();
}

class SdyCustomCallPattern : public OpConversionPattern<CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getCallTargetName() == kShardingCustomCallTargetName) {
      return rewriteShardingCustomCall(op, adaptor, rewriter);
    }

    if (op.getCallTargetName() == kShardingGroupCustomCallTargetName) {
      return rewriteShardingGroupCustomCall(op, adaptor, rewriter);
    }

    return rewriter.notifyMatchFailure(
        op, "expected CustomCallOp with xla.sdy target name.");
  }
};

// Convert custom calls into sdy APIs.
// * xla.sdy.Sharding -> ShardingConstraintOp
// * xla.sdy.ShardingGroup -> ShardingGroupOp
class ImportSdyCustomCallsPass
    : public PassWrapper<ImportSdyCustomCallsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportSdyCustomCallsPass)

  void runOnOperation() final {
    MLIRContext& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<SdyDialect>();
    target.addDynamicallyLegalOp<CustomCallOp>([](CustomCallOp op) {
      return op.getCallTargetName() != kShardingCustomCallTargetName &&
             op.getCallTargetName() != kShardingGroupCustomCallTargetName;
    });
    RewritePatternSet patterns(&context);
    patterns.add<SdyCustomCallPattern>(&context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "sdy-import-sdy-custom-calls";
  }

  StringRef getDescription() const override {
    return "Converts a CustomCall with target name Sharding into a "
           "ShardingConstraintOp and with target name ShardingGroup into a "
           "ShardingGroupOp.";
  }
};

}  // namespace

std::unique_ptr<Pass> createImportSdyCustomCallsPass() {
  return std::make_unique<ImportSdyCustomCallsPass>();
}

void registerImportSdyCustomCallsPass() {
  registerPass(createImportSdyCustomCallsPass);
}

}  // namespace sdy
}  // namespace mlir
