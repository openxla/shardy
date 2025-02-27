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

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
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

namespace sdy {
namespace round_trip_import {

namespace {

using ::mlir::IntegerAttr;
using ::mlir::StringRef;
using ::mlir::sdy::ShardingConstraintOp;
using ::mlir::sdy::ShardingGroupOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::stablehlo::CustomCallOp;
using ::mlir::stablehlo::CustomCallOpAdaptor;

mlir::LogicalResult rewriteShardingCustomCall(
    CustomCallOp op, CustomCallOpAdaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) {
  std::vector<int64_t> unspecDims;
  if (std::optional<mlir::Attribute> backendConfig = op.getBackendConfig()) {
    StringRef str = mlir::dyn_cast<mlir::StringAttr>(*backendConfig).getValue();
    if (!str.empty()) {
      if (!str.starts_with("unspecified_dims=[")) {
        return op.emitError()
               << "expected sharding attribute starts with unspecified_dims=[";
      }
      str = str.drop_front(std::strlen("unspecified_dims=["));
      if (!str.ends_with("]")) {
        return op.emitError() << "expected sharding attribute ends with ]";
      }
      StringRef rhs = str.drop_back(std::strlen("]"));
      StringRef lhs;
      while (!rhs.empty()) {
        if (rhs.starts_with(" ")) {
          rhs = rhs.drop_front(1);
        }
        std::tie(lhs, rhs) = rhs.split(',');
        int64_t dim_int;
        if (!lhs.getAsInteger(10, dim_int)) {
          unspecDims.push_back(dim_int);
        } else {
          return op.emitError() << "expected sharding attribute to be a comma "
                                   "separated list of integers";
        }
      }
    }
  }

  if (op->getNumResults() != 1) {
    op.emitError() << "expected CustomCallOp with exactly one result";
    return mlir::failure();
  }
  TensorShardingAttr sharding = mlir::sdy::getSharding(op->getResult(0));
  if (!sharding) {
    op.emitError() << "expected CustomCallOp with a sharding attribute";
    return mlir::failure();
  }

  if (!unspecDims.empty()) {
    sharding = sharding.openShardingDims(unspecDims);
  }

  rewriter.replaceOpWithNewOp<ShardingConstraintOp>(
      op, adaptor.getInputs().front(), sharding);

  return mlir::success();
}

mlir::LogicalResult rewriteShardingGroupCustomCall(
    CustomCallOp op, CustomCallOpAdaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) {
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

  return mlir::success();
}

class SdyCustomCallPattern : public mlir::OpConversionPattern<CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
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
    : public mlir::PassWrapper<ImportSdyCustomCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportSdyCustomCallsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addLegalDialect<mlir::sdy::SdyDialect>();
    target.addDynamicallyLegalOp<CustomCallOp>([](CustomCallOp op) {
      return op.getCallTargetName() != kShardingCustomCallTargetName &&
             op.getCallTargetName() != kShardingGroupCustomCallTargetName;
    });
    mlir::RewritePatternSet patterns(&context);
    patterns.add<SdyCustomCallPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-import-sdy-custom-calls";
  }

  StringRef getDescription() const override {
    return "Converts a CustomCall with target name Sharding into a "
           "ShardingConstraintOp and with target name ShardingGroup into a "
           "ShardingGroupOp.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportSdyCustomCallsPass() {
  return std::make_unique<ImportSdyCustomCallsPass>();
}

void registerImportSdyCustomCallsPass() {
  mlir::registerPass(createImportSdyCustomCallsPass);
}

}  // namespace round_trip_import
}  // namespace sdy
