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
// constants (exist in `constantOpsUsed`).
bool isConstantExpression(
    Operation* op, const llvm::DenseMap<Operation*, bool>& constantOpsUsed) {
  if (isa<ConstantOp, stablehlo::IotaOp>(op)) {
    return true;
  }
  return (isa<stablehlo::BroadcastInDimOp, stablehlo::SliceOp>(op) ||
          isElementwise(op)) &&
         isPure(op) && llvm::all_of(op->getOperands(), [&](Value operand) {
           return operand.getDefiningOp() &&
                  constantOpsUsed.contains(operand.getDefiningOp());
         });
}

bool isConstantOrItsPredecessorsUsed(
    Operation* op, const llvm::DenseMap<Operation*, bool>& constantOpsUsed) {
  if (constantOpsUsed.at(op)) {
    return true;
  }
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    if (auto defOpResult = dyn_cast<OpResult>(operand)) {
      return isConstantOrItsPredecessorsUsed(defOpResult.getOwner(),
                                             constantOpsUsed);
    }
    return false;
  });
}

void markConstantAndItsPredecessorsUsed(
    Operation* op, llvm::DenseMap<Operation*, bool>& constantOpsUsed) {
  constantOpsUsed[op] = true;
  for (Value operand : op->getOperands()) {
    if (auto defOpResult = dyn_cast<OpResult>(operand)) {
      markConstantAndItsPredecessorsUsed(defOpResult.getOwner(),
                                         constantOpsUsed);
    }
  }
}

// If the given op is already in `mapping`, returns the mapped value.
//
// Otherwise, apply this function recursively on all operands. And then,
// * If the op is used, clone the op and replace the operand with the cloned op.
// * If the op is not used, replace the operand with the original op.
Value cloneSubComputation(
    OpResult opResult, IRMapping& mapping,
    const llvm::DenseMap<Operation*, bool>& constantOpsUsed) {
  if (mapping.lookupOrNull(opResult)) {
    return mapping.lookup(opResult);
  }

  Operation* op = opResult.getOwner();
  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands()) {
    if (auto defOpResult = dyn_cast<OpResult>(operand)) {
      newOperands.push_back(
          cloneSubComputation(defOpResult, mapping, constantOpsUsed));
    } else {
      newOperands.push_back(operand);
    }
  }

  if (constantOpsUsed.at(op)) {
    // This will insert the cloned op right before the original op.
    OpBuilder builder(op);
    builder.clone(*op, mapping);
    cloneShardingGroupUsers(opResult, mapping, builder);
    return mapping.lookup(opResult);
  }

  op->setOperands(newOperands);
  return opResult;
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
    llvm::DenseMap<Operation*, bool> constantOpsUsed;
    funcOp.walk([&](Operation* op) {
      if (isa<ShardingGroupOp>(op)) {
        return;
      }
      if (isConstantExpression(op, constantOpsUsed)) {
        constantOpsUsed[op] = false;
        return;
      }

      for (OpOperand& operand : op->getOpOperands()) {
        if (auto defOpResult = dyn_cast<OpResult>(operand.get());
            defOpResult && constantOpsUsed.contains(defOpResult.getOwner())) {
          // `op` is not a constant expression, but its `operand` is a constant
          // expression.
          if (isConstantOrItsPredecessorsUsed(defOpResult.getOwner(),
                                              constantOpsUsed)) {
            // If this operand or its predecessors are marked as used, we need
            // to clone the used constant sub-computation.
            IRMapping mapping;
            operand.set(
                cloneSubComputation(defOpResult, mapping, constantOpsUsed));
          }
          // Mark the constant and its predecessors as used, so that they cannot
          // be used by other non-constant expressions.
          markConstantAndItsPredecessorsUsed(defOpResult.getOwner(),
                                             constantOpsUsed);
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
