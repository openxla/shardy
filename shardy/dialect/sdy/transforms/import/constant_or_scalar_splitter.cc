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
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/op_properties.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_CONSTANTORSCALARSPLITTERPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

void cloneShardingGroupUsers(OpResult opResult, IRMapping& mapping,
                             OpBuilder& builder) {
  for (Operation* user : opResult.getUsers()) {
    if (isa<ShardingGroupOp>(user)) {
      builder.clone(*user, mapping);
    }
  }
}

// A constant preserving op is an op that is considered a constant expression if
// it is pure and all its results can be considered as constant expressions
// given all its operands are constant expressions, for which it holds if the
// given op is either:
// - A broadcast, reshape or slice op.
// - An elementwise op.
// - A named computation all operations are constant preserving.
// - A call to a func that all operations are constant preserving.
// Assumes the op is not constant or iota.
bool isConstantPreserving(
    Operation* op,
    const llvm::SmallDenseSet<StringRef>& nonConstantNamedComputationOps,
    const llvm::SmallDenseSet<StringRef>& nonConstFuncOps,
    const SymbolTable& symbolTable) {
  if (CallOp callOp = dyn_cast<CallOp>(op)) {
    return !nonConstFuncOps.contains(getOriginalFuncName(callOp, symbolTable));
  }
  if (auto namedComputationOp = dyn_cast<NamedComputationOp>(op)) {
    return !nonConstantNamedComputationOps.contains(
        namedComputationOp.getName());
  }
  if (!isPure(op)) {
    return false;
  }
  if (isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
          stablehlo::SliceOp>(op)) {
    return true;
  }
  if (isElementwise(op)) {
    return true;
  }
  return false;
}

// Returns true if the given op is either:
// - A constant or iota op.
// - A constant preserving op. (see isConstantPreserving) and all operands are
// constants, that is, exist in `constantOps`.
bool isConstantExpression(
    Operation* op, const llvm::SetVector<Operation*>& constantOps,
    const llvm::SmallDenseSet<StringRef>& nonConstantNamedComputationOps,
    const llvm::SmallDenseSet<StringRef>& nonConstFuncOps,
    const SymbolTable& symbolTable) {
  if (isa<ConstantOp, stablehlo::IotaOp>(op)) {
    return true;
  }
  return isConstantPreserving(op, nonConstantNamedComputationOps,
                              nonConstFuncOps, symbolTable) &&
         llvm::all_of(op->getOperands(), [&](Value operand) {
           return operand.getDefiningOp() &&
                  constantOps.contains(operand.getDefiningOp());
         });
}

// Returns true if the given op is a broadcast of scalar.
bool isScalarExpansion(Operation* op) {
  // TODO(enver): Allow for any tensor with exactly one element.
  if (auto broadcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op);
      broadcastOp && isScalar(broadcastOp.getOperand())) {
    return true;
  }
  return false;
}

// Recursively clones all operands of the given op, that are not already mapped
// in `mapping`, and finally clones the op itself. We do not clone scalars as
// they do not get sharded.
void cloneSubComputation(OpResult opResult, IRMapping& mapping,
                         SymbolTable& symbolTable) {
  if (isScalar(opResult) || mapping.lookupOrNull(opResult)) {
    return;
  }
  Operation* op = opResult.getOwner();
  for (Value operand : op->getOperands()) {
    if (auto defOpResult = dyn_cast<OpResult>(operand)) {
      cloneSubComputation(defOpResult, mapping, symbolTable);
    }
  }

  // This will insert the cloned op right before the original op.
  OpBuilder builder(op);
  Operation* clonedOp = builder.clone(*op, mapping);
  if (CallOp callOp = dyn_cast<CallOp>(clonedOp)) {
    FuncOp funcOp = symbolTable.lookup<FuncOp>(callOp.getCallee());
    callOp.setCallee(
        symbolTable.insert(cloneFuncRecursively(funcOp, symbolTable)));
  }
  cloneShardingGroupUsers(opResult, mapping, builder);
}

// Recursively clones all operands of the given op, that are not already cloned,
// and finally clones the op itself. We do not clone scalars as they do not get
// sharded.
//
// Returns the cloned op result.
Value cloneSubComputation(OpResult opResult, SymbolTable& symbolTable) {
  if (isScalar(opResult)) {
    return opResult;
  }
  IRMapping mapping;
  cloneSubComputation(opResult, mapping, symbolTable);
  return mapping.lookup(opResult);
}

void cloneSubComputationOnOperands(
    Operation* op, const llvm::SetVector<Operation*>& constantOps,
    const llvm::SetVector<Operation*>& scalarExpansionOps,
    SymbolTable& symbolTable) {
  for (OpOperand& operand : op->getOpOperands()) {
    if (auto defOpResult = dyn_cast<OpResult>(operand.get());
        defOpResult && (constantOps.contains(defOpResult.getOwner()) ||
                        scalarExpansionOps.contains(defOpResult.getOwner()))) {
      // `op` is not a constant expression, while its `operand` is. We
      // recursively clone the sub-computation whose root is
      // `defOpResult`, and replace the `operand` with the cloned defining
      // op. The cloned constant sub-computation has only one user `op`,
      // so that it is isolated from the rest of the computation.
      operand.set(cloneSubComputation(defOpResult, symbolTable));
    }
  }
}

void processOp(Operation* op, FuncOp funcOp,
               llvm::SetVector<Operation*>& constantOps,
               llvm::SetVector<Operation*>& scalarExpansionOps,
               llvm::SmallDenseSet<StringRef>& nonConstantNamedComputationOps,
               llvm::SmallDenseSet<StringRef>& nonConstFuncOps,
               SymbolTable& symbolTable) {
  if (isa<FuncOp, ShardingGroupOp>(op)) {
    return;
  }
  if (isConstantExpression(op, constantOps, nonConstantNamedComputationOps,
                           nonConstFuncOps, symbolTable)) {
    constantOps.insert(op);
    return;
  }
  // NOTE: There are cases that op is an constant expression but may not pass
  // the following check such as constant and iota ops. That is fine because if
  // the op is a constant expression it is a stronger condition than being just
  // constant preserving and it does not make the parent named computation or
  // the `funcOp` non-const, and at this point, it is guaranteed that the op is
  // not constant expression.
  if (!isConstantPreserving(op, nonConstantNamedComputationOps, nonConstFuncOps,
                            symbolTable) &&
      !op->hasTrait<OpTrait::IsTerminator>()) {
    if (auto namedCompuationOp = op->getParentOfType<NamedComputationOp>()) {
      nonConstantNamedComputationOps.insert(namedCompuationOp.getName());
    }
    nonConstFuncOps.insert(getOriginalFuncName(funcOp));
  }
  if (isScalarExpansion(op)) {
    scalarExpansionOps.insert(op);
    return;
  }
  cloneSubComputationOnOperands(op, constantOps, scalarExpansionOps,
                                symbolTable);
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

struct ConstantOrScalarSplitterPass
    : public impl::ConstantOrScalarSplitterPassBase<
          ConstantOrScalarSplitterPass> {
  using ConstantOrScalarSplitterPassBase::ConstantOrScalarSplitterPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<stablehlo::ConstantOp>();
    target->addLegalOp<ConstantOp>();

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<ConstantPattern>(context);
    patterns = std::move(patternsInternal);

    return success();
  }

  template <typename RangeT>
  void eraseUnusedOpsAlongWithItsShardingGroupUsers(RangeT&& ops) {
    for (Operation* op : ops) {
      if (!hasOnlyUsersOfType<ShardingGroupOp>(op)) {
        continue;
      }
      // All users are ShardingGroupOps. Erase them first.
      for (Operation* user : llvm::make_early_inc_range(op->getUsers())) {
        user->erase();
      }
      op->erase();
    }
  }

  void walkOnRegion(
      mlir::Region& region, FuncOp funcOp,
      llvm::SmallDenseSet<StringRef>& nonConstantNamedComputationOps,
      llvm::SmallDenseSet<StringRef>& nonConstFuncOps,
      SymbolTable& symbolTable) {
    llvm::SetVector<Operation*> constantOps;
    llvm::SetVector<Operation*> scalarExpansionOps;
    region.walk<WalkOrder::PreOrder>([&](Operation* op) {
      processOp(op, funcOp, constantOps, scalarExpansionOps,
                nonConstantNamedComputationOps, nonConstFuncOps, symbolTable);
      // Skip walking on the `NamedComputationOp`.
      if (isa<NamedComputationOp>(op)) {
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    // Since for every op in `constantOps` that has a use that isn't in
    // `constantOps`, we replaced the use with a clone of the entire
    // sub-computation, we can now erase all ops in `constantOps` as long as we
    // iterate in reverse order. Note that we did not clone scalars so we keep
    // the original.
    eraseUnusedOpsAlongWithItsShardingGroupUsers(llvm::concat<Operation* const>(
        scalarExpansionOps, llvm::reverse(constantOps)));
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // We first convert any `stablehlo::ConstantOp` to an `sdy::ConstantOp`, so
    // that constants won't be deduped via folding.
    if (failed(applyPartialConversion(moduleOp, *target, patterns))) {
      signalPassFailure();
    }

    // Then we split constant sub-computations for each non-constant user.
    llvm::SmallDenseSet<StringRef> nonConstantNamedComputationOps;
    llvm::SmallDenseSet<StringRef> nonConstFuncOps;
    iterateFuncs(moduleOp, [&](FuncOp funcOp) {
      funcOp.walk([&](NamedComputationOp namedComputationOp) {
        walkOnRegion(namedComputationOp.getBody(), funcOp,
                     nonConstantNamedComputationOps, nonConstFuncOps,
                     symbolTable);
      });
      walkOnRegion(funcOp.getBody(), funcOp, nonConstantNamedComputationOps,
                   nonConstFuncOps, symbolTable);
    });
  }

 private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
