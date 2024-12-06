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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/op_properties.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_CONSTANTSPLITTERPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using func::FuncOp;

// Returns true if `op` has multiple uses that are not by `ShardingGroupOp`.
bool hasMultipleRealUses(Operation* op) {
  bool seenUse = false;
  for (Operation* user : op->getUsers()) {
    if (!isa<ShardingGroupOp>(user)) {
      if (seenUse) {
        return true;
      }
      seenUse = true;
    }
  }
  return false;
}

void cloneShardingGroupUsers(OpResult opResult, IRMapping& mapping,
                             OpBuilder& builder) {
  for (Operation* user : opResult.getUsers()) {
    if (auto shardingGroupOp = dyn_cast<ShardingGroupOp>(user)) {
      builder.clone(*shardingGroupOp, mapping);
    }
  }
}

// Returns true if the given op is either:
// - A constant or iota op.
// - A broadcast, slice, or pure element-wise op whose operands are all
// constants (exist in `constantOps`).
bool isConstantExpression(Operation* op,
                          const llvm::DenseSet<Operation*>& constantOps) {
  if (isa<ConstantOp, stablehlo::IotaOp>(op)) {
    return true;
  }
  return (isa<stablehlo::BroadcastInDimOp, stablehlo::SliceOp>(op) ||
          isElementwise(op)) &&
         isPure(op) && llvm::all_of(op->getOperands(), [&](Value operand) {
           return operand.getDefiningOp() &&
                  constantOps.contains(operand.getDefiningOp());
         });
}

// Recursively clones all operands of the given op, that are not already mapped
// in `mapping`, and finally clones the op itself.
void cloneSubComputation(OpResult opResult, IRMapping& mapping) {
  Operation* op = opResult.getOwner();
  for (Value operand : op->getOperands()) {
    if (auto defOpResult = dyn_cast<OpResult>(operand)) {
      cloneSubComputation(defOpResult, mapping);
    }
  }

  if (!mapping.lookupOrNull(opResult)) {
    // This will insert the cloned op right before the original op.
    OpBuilder builder(op);
    builder.clone(*op, mapping);
    cloneShardingGroupUsers(opResult, mapping, builder);
  }
}

// Recursively clones all operands of the given op, that are not already cloned,
// and finally clones the op itself.
//
// Returns the cloned op result.
Value cloneSubComputation(OpResult opResult) {
  IRMapping mapping;
  cloneSubComputation(opResult, mapping);
  return mapping.lookup(opResult);
}

// Converts stablehlo::ConstantOp to sdy::ConstantOp.
class ConstantPattern : public OpConversionPattern<stablehlo::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // We use the generic op builder so that unregistered attributes will be
    // added to the new op.
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, op->getResultTypes(), adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

struct ConstantSplitterPass
    : public impl::ConstantSplitterPassBase<ConstantSplitterPass> {
  using ConstantSplitterPassBase::ConstantSplitterPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<stablehlo::ConstantOp>();
    target->addLegalOp<ConstantOp>();

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<ConstantPattern>(context);
    patterns = std::move(patternsInternal);

    return success();
  }

  void runOnOperation() final {
    FuncOp funcOp = getOperation();

    // We first convert any `stablehlo::ConstantOp` to an `sdy::ConstantOp`, so
    // that constants won't be deduped via folding.
    if (failed(applyPartialConversion(funcOp, *target, patterns))) {
      signalPassFailure();
    }

    // Then we split constant sub-computations for each non-constant user.
    llvm::DenseSet<Operation*> constantOps;
    funcOp.walk([&](Operation* op) {
      if (!isConstantExpression(op, constantOps) || isa<ShardingGroupOp>(op)) {
        return;
      }

      // `op` is a constant expression.
      constantOps.insert(op);

      if (!hasMultipleRealUses(op)) {
        // No need to split `op`.
        return;
      }

      // For each use of `op`, we recursively clone the sub-computation whose
      // root is `op`, and replace the use with the cloned defining op. This
      // will ensure that by the end of this walk, all constant sub-computations
      // will have a single user.
      for (OpOperand& use : op->getUses()) {
        if (!isa<ShardingGroupOp>(use.getOwner())) {
          use.set(cloneSubComputation(cast<OpResult>(use.get())));
        }
      }
    });
  }

 private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
