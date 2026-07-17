/* Copyright 2026 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_VERIFYUNREDUCEDAXESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Appends a sharding's unreduced axes to the given vector.
void appendUnreducedAxes(TensorShardingAttr sharding,
                         SmallVectorImpl<AxisRefAttr>& axes) {
  if (sharding) {
    llvm::append_range(axes, sharding.getUnreducedAxes());
  }
}

// Verifies that all unreduced axes in `sourceAxes` are present in `targetAxes`.
// Otherwise, emits an op error with `errorMsgSuffix` and returns failure.
LogicalResult verifySubset(ArrayRef<AxisRefAttr> sourceAxes,
                           ArrayRef<AxisRefAttr> targetAxes, Operation* op,
                           const llvm::Twine& errorMsgSuffix) {
  for (AxisRefAttr axis : sourceAxes) {
    if (!llvm::is_contained(targetAxes, axis)) {
      return op->emitOpError("dropped unreduced axis '")
             << axis.getName() << errorMsgSuffix;
    }
  }
  return success();
}

// Verifies that `sourceAxes` and `targetAxes` contain the same unreduced axes.
// Otherwise, emits an op error with `errorMsgSuffix` and returns failure.
LogicalResult verifyEqual(ArrayRef<AxisRefAttr> sourceAxes,
                          ArrayRef<AxisRefAttr> targetAxes, Operation* op,
                          const llvm::Twine& errorMsgSuffix) {
  if (sourceAxes != targetAxes) {
    // Generate mismatch names to keep the error message helpful.
    SmallVector<StringRef> mismatchNames;
    for (AxisRefAttr axis : sourceAxes) {
      if (!llvm::is_contained(targetAxes, axis)) {
        mismatchNames.push_back(axis.getName());
      }
    }
    for (AxisRefAttr axis : targetAxes) {
      if (!llvm::is_contained(sourceAxes, axis)) {
        mismatchNames.push_back(axis.getName());
      }
    }
    llvm::sort(mismatchNames);
    mismatchNames.erase(llvm::unique(mismatchNames), mismatchNames.end());
    std::string joinedNames = llvm::join(mismatchNames, ", ");
    return op->emitOpError("has unreduced axes mismatch for '")
           << joinedNames << errorMsgSuffix;
  }
  return success();
}

bool isDefiningOpBlessedOrDot(Value value) {
  if (!value) {
    return false;
  }
  Operation* defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (isa<sdy::ReshardOp, sdy::ShardingConstraintOp, stablehlo::DotGeneralOp,
          stablehlo::DotOp>(defOp)) {
    return true;
  }
  StringRef opName = defOp->getName().getStringRef();
  return opName == "mhlo.dot_general" || opName == "mhlo.dot" ||
         opName == "stablehlo.copy" || opName == "mhlo.copy";
}

// Verifies that a call operation has the exact same unreduced axes at its
// operand-argument boundary and callee-caller result boundary.
LogicalResult verifyCallUnreducedAxes(func::CallOp callOp) {
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
  if (!callee) {
    return success();
  }

  // Verify call operands against callee arguments.
  for (int64_t i = 0; i < callOp.getNumOperands(); ++i) {
    Value operand = callOp.getOperand(i);
    if (isDefiningOpBlessedOrDot(operand)) {
      continue;
    }
    TensorShardingAttr operandSharding = getShardingBypassingBarriers(operand);
    TensorShardingAttr argSharding =
        getShardingBypassingBarriers(callee.getArgument(i));
    ArrayRef<AxisRefAttr> operandAxes = operandSharding
                                            ? operandSharding.getUnreducedAxes()
                                            : ArrayRef<AxisRefAttr>{};
    ArrayRef<AxisRefAttr> argAxes =
        argSharding ? argSharding.getUnreducedAxes() : ArrayRef<AxisRefAttr>{};
    if (failed(verifyEqual(
            operandAxes, argAxes, callOp,
            llvm::Twine("' at call argument ") + llvm::Twine(i) + "."))) {
      return failure();
    }
  }

  // Verify callee results against call results.
  for (int64_t i = 0; i < callOp.getNumResults(); ++i) {
    TensorShardingAttr funcResultSharding = getFuncResultSharding(callee, i);
    TensorShardingAttr resultSharding =
        getShardingBypassingBarriers(callOp.getResult(i));
    ArrayRef<AxisRefAttr> funcResultAxes =
        funcResultSharding ? funcResultSharding.getUnreducedAxes()
                           : ArrayRef<AxisRefAttr>{};
    ArrayRef<AxisRefAttr> resultAxes = resultSharding
                                           ? resultSharding.getUnreducedAxes()
                                           : ArrayRef<AxisRefAttr>{};
    if (failed(verifyEqual(
            funcResultAxes, resultAxes, callOp,
            llvm::Twine("' at call result ") + llvm::Twine(i) + "."))) {
      return failure();
    }
  }

  return success();
}

LogicalResult verifyReshardUnreducedAxes(Operation* op) {
  TensorShardingAttr sourceSharding =
      getShardingBypassingBarriers(op->getOperand(0));
  TensorShardingAttr targetSharding =
      op->getAttrOfType<TensorShardingAttr>("sharding");
  if (!sourceSharding || !targetSharding) {
    return success();
  }
  bool hasKeptAxes = false;
  for (AxisRefAttr targetAxis : targetSharding.getUnreducedAxes()) {
    if (llvm::is_contained(sourceSharding.getUnreducedAxes(), targetAxis)) {
      hasKeptAxes = true;
      break;
    }
  }
  if (hasKeptAxes &&
      sourceSharding.getReductionOp() != targetSharding.getReductionOp()) {
    return op->emitOpError(
               "cannot change the reduction operator of kept unreduced axes "
               "from ")
           << sourceSharding.getReductionOp() << " to "
           << targetSharding.getReductionOp() << ". Check source sharding: "
           << sourceSharding << ", target sharding: " << targetSharding;
  }
  return success();
}

// Verifies that standard operations and ReturnLike terminators do not drop
// or introduce unreduced axes.
LogicalResult verifyDefaultOpUnreducedAxes(Operation* op, func::FuncOp funcOp) {
  SmallVector<AxisRefAttr> operandUnreducedAxes;
  SmallVector<AxisRefAttr> resultUnreducedAxes;

  if (op->hasTrait<OpTrait::ReturnLike>() &&
      isa<func::FuncOp>(op->getParentOp())) {
    auto parentFuncOp = cast<func::FuncOp>(op->getParentOp());
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      Value operand = op->getOperand(i);
      if (isDefiningOpBlessedOrDot(operand)) {
        continue;
      }
      TensorShardingAttr resultSharding =
          getFuncResultSharding(parentFuncOp, i);
      if (!resultSharding) {
        continue;
      }
      appendUnreducedAxes(getShardingBypassingBarriers(operand),
                          operandUnreducedAxes);
      appendUnreducedAxes(resultSharding, resultUnreducedAxes);
    }

    if (operandUnreducedAxes.empty() && resultUnreducedAxes.empty()) {
      return success();
    }
    return verifyEqual(
        operandUnreducedAxes, resultUnreducedAxes, op,
        "' without a blessed operation (e.g., sdy.reshard). "
        "This is an invalid transition from unreduced to reduced.");
  }

  for (Value operand : op->getOperands()) {
    if (isDefiningOpBlessedOrDot(operand)) {
      continue;
    }
    appendUnreducedAxes(getShardingBypassingBarriers(operand),
                        operandUnreducedAxes);
  }

  if (op->hasTrait<OpTrait::ReturnLike>()) {
    Operation* parentOp = op->getParentOp();
    if (parentOp->getNumResults() > 0) {
      for (int64_t i = 0; i < op->getNumOperands(); ++i) {
        if (i < parentOp->getNumResults()) {
          TensorShardingAttr resultSharding =
              getShardingBypassingBarriers(parentOp->getResult(i));
          if (auto manualCompOp =
                  dyn_cast<sdy::ManualComputationOp>(parentOp)) {
            resultSharding =
                eraseManualAxes(resultSharding, manualCompOp.getManualAxes());
          }
          appendUnreducedAxes(resultSharding, resultUnreducedAxes);
        }
      }
    }

    if (operandUnreducedAxes.empty() && resultUnreducedAxes.empty()) {
      return success();
    }
    return verifyEqual(
        operandUnreducedAxes, resultUnreducedAxes, op,
        "' without a blessed operation (e.g., sdy.reshard). "
        "This is an invalid transition from unreduced to reduced.");
  }

  for (Value result : op->getResults()) {
    appendUnreducedAxes(getShardingBypassingBarriers(result),
                        resultUnreducedAxes);
  }

  if (operandUnreducedAxes.empty()) {
    return success();
  }

  // Operations like stablehlo.dot_general can legitimately introduce new
  // unreduced axes on their results. Instead of coding a fragile whitelist of
  // such operations, we just check that all unreduced axes in the operands
  // are still present in the results (and not dropped without a reeshard).
  return verifySubset(
      operandUnreducedAxes, resultUnreducedAxes, op,
      "' without a blessed operation (e.g., sdy.reshard). "
      "This is an invalid transition from unreduced to reduced.");
}

bool isSumReduction(stablehlo::ReduceOp reduceOp) {
  if (reduceOp.getBody().empty()) {
    return false;
  }
  Block& body = reduceOp.getBody().front();
  auto returnOp = dyn_cast<stablehlo::ReturnOp>(body.getTerminator());
  if (!returnOp ||
      returnOp.getOperands().size() != reduceOp.getInputs().size()) {
    return false;
  }
  for (Value retValue : returnOp.getOperands()) {
    Operation* curOp = retValue.getDefiningOp();
    if (!curOp || !isa<stablehlo::AddOp>(curOp)) {
      return false;
    }
  }
  return true;
}

struct VerifyUnreducedAxesPass
    : public impl::VerifyUnreducedAxesPassBase<VerifyUnreducedAxesPass> {
  using VerifyUnreducedAxesPassBase::VerifyUnreducedAxesPassBase;

 protected:
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();

    auto walkResult = funcOp.walk([&](Operation* op) {
      if (auto reshardOp = dyn_cast<sdy::ReshardOp>(op)) {
        if (failed(verifyReshardUnreducedAxes(reshardOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto constraintOp = dyn_cast<sdy::ShardingConstraintOp>(op)) {
        if (failed(verifyReshardUnreducedAxes(constraintOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (isa<stablehlo::DotGeneralOp, stablehlo::DotOp, sdy::AllReduceOp,
              sdy::ReduceScatterOp>(op)) {
        return WalkResult::advance();
      }

      if (auto reduceOp = dyn_cast<stablehlo::ReduceOp>(op)) {
        if (isSumReduction(reduceOp)) {
          return WalkResult::advance();
        }
      }

      if (op->hasTrait<OpTrait::IsTerminator>() &&
          !op->hasTrait<OpTrait::ReturnLike>()) {
        return WalkResult::advance();
      }

      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (failed(verifyCallUnreducedAxes(callOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (failed(verifyDefaultOpUnreducedAxes(op, funcOp))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
