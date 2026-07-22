/* Copyright 2026 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
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


// Verifies that `sourceAxes` and `targetAxes` contain the same unreduced axes.
// Otherwise, emits an op error with `errorMsgSuffix` and returns failure.
LogicalResult verifyEqual(ArrayRef<AxisRefAttr> sourceAxes,
                          ArrayRef<AxisRefAttr> targetAxes, Operation* op,
                          const llvm::Twine& errorMsgSuffix) {
  SmallVector<AxisRefAttr> sortedSource(sourceAxes);
  SmallVector<AxisRefAttr> sortedTarget(targetAxes);
  llvm::sort(sortedSource);
  sortedSource.erase(llvm::unique(sortedSource), sortedSource.end());
  llvm::sort(sortedTarget);
  sortedTarget.erase(llvm::unique(sortedTarget), sortedTarget.end());

  if (sortedSource != sortedTarget) {
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

bool isExplicitReshardOp(Operation* op) {
  return isa_and_nonnull<sdy::ReshardOp, sdy::ShardingConstraintOp>(op);
}

bool isDefiningOpExplicitReshard(Value value) {
  if (!value) {
    return false;
  }
  while (auto barrierOp = value.getDefiningOp<PropagationBarrierOp>()) {
    value = barrierOp.getInput();
  }
  return isExplicitReshardOp(value.getDefiningOp());
}

/// Verifies that a call operation has the exact same unreduced axes at its
// operand-argument boundary and callee-caller result boundary.
LogicalResult verifyCall(func::CallOp callOp) {
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
  if (!callee) {
    return success();
  }

  // Verify call operands against callee arguments.
  for (int64_t i = 0; i < callOp.getNumOperands(); ++i) {
    Value operand = callOp.getOperand(i);
    if (isDefiningOpExplicitReshard(operand)) {
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

LogicalResult verifyReturnPair(Value operand, TensorShardingAttr resultSharding,
                               Operation* op, int64_t index) {
  if (isDefiningOpExplicitReshard(operand)) {
    return success();
  }
  if (!resultSharding) {
    return success();
  }
  SmallVector<AxisRefAttr> operandAxes;
  SmallVector<AxisRefAttr> resultAxes;
  appendUnreducedAxes(getShardingBypassingBarriers(operand), operandAxes);
  appendUnreducedAxes(resultSharding, resultAxes);
  return verifyEqual(operandAxes, resultAxes, op,
                     llvm::Twine("' at return value ") + llvm::Twine(index) +
                         " without a blessed operation (e.g., sdy.reshard). "
                         "This is an invalid transition "
                         "from unreduced to reduced.");
}

// Verifies that standard operations and ReturnLike terminators do not drop
// or introduce unreduced axes.
LogicalResult verifyTerminator(Operation* op, func::FuncOp funcOp) {
  // Verify function-level terminator: the return operands must match the
  // function results.
  if (isa<func::FuncOp>(op->getParentOp())) {
    auto parentFuncOp = cast<func::FuncOp>(op->getParentOp());
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      TensorShardingAttr resultSharding =
          getFuncResultSharding(parentFuncOp, i);
      if (failed(verifyReturnPair(op->getOperand(i), resultSharding, op, i))) {
        return failure();
      }
    }
    return success();
  }

  // Verify nested terminator: the return operands must match the manual
  // computation results.
  Operation* parentOp = op->getParentOp();
  if (parentOp->getNumResults() > 0) {
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      if (i < parentOp->getNumResults()) {
        TensorShardingAttr resultSharding =
            getShardingBypassingBarriers(parentOp->getResult(i));
        if (auto manualCompOp = dyn_cast<sdy::ManualComputationOp>(parentOp)) {
          resultSharding =
              eraseManualAxes(resultSharding, manualCompOp.getManualAxes());
        }
        if (failed(
                verifyReturnPair(op->getOperand(i), resultSharding, op, i))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult verifyDefaultOp(Operation* op, func::FuncOp funcOp) {
  SmallVector<TensorShardingAttr> operandShardings;
  operandShardings.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    operandShardings.push_back(isDefiningOpExplicitReshard(operand)
                                   ? TensorShardingAttr()
                                   : getShardingBypassingBarriers(operand));
  }
  SmallVector<TensorShardingAttr> resultShardings;
  resultShardings.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    resultShardings.push_back(getShardingBypassingBarriers(result));
  }
  // We currently allow default ops to introduce SUM unreduced axes.
  return verifyUnreducedAxesTransition(
      op, operandShardings, resultShardings,
      /*expectedIntroducedRedOp=*/ReductionOp::SUM);
}

std::optional<ReductionOp> getReductionType(Region& region) {
  if (region.empty()) {
    return std::nullopt;
  }
  Block& body = region.front();
  auto returnOp = dyn_cast<stablehlo::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getOperands().empty()) {
    return std::nullopt;
  }
  std::optional<ReductionOp> result;
  for (Value operand : returnOp.getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (!defOp) {
      return std::nullopt;
    }
    std::optional<ReductionOp> curOpType;
    if (isa<stablehlo::AddOp>(defOp)) {
      curOpType = ReductionOp::SUM;
    } else if (isa<stablehlo::MaxOp>(defOp)) {
      curOpType = ReductionOp::MAX;
    } else if (isa<stablehlo::MinOp>(defOp)) {
      curOpType = ReductionOp::MIN;
    } else {
      return std::nullopt;
    }
    if (!result) {
      result = curOpType;
    } else if (result != curOpType) {
      return std::nullopt;
    }
  }
  return result;
}

LogicalResult verifyReduce(stablehlo::ReduceOp reduceOp) {
  SmallVector<TensorShardingAttr> operandShardings;
  for (Value operand : reduceOp.getInputs()) {
    operandShardings.push_back(isDefiningOpExplicitReshard(operand)
                                   ? TensorShardingAttr()
                                   : getShardingBypassingBarriers(operand));
  }
  SmallVector<TensorShardingAttr> resultShardings;
  for (Value result : reduceOp.getResults()) {
    resultShardings.push_back(getShardingBypassingBarriers(result));
  }

  std::optional<ReductionOp> expectedBodyRedOp =
      getReductionType(reduceOp.getBody());

  return verifyUnreducedAxesTransition(reduceOp, operandShardings,
                                       resultShardings, expectedBodyRedOp);
}

// We currently verify this for dot and dot_general without analyzing the
// semantics of the operation:
// 1. Unreduced axes from the operands can be passed through to the results
//    (such as for the batching dimensions of the op).
// 2. Unreduced axes introduced by the result can only be SUM reduction
//    (such as for the partitioned contraction dimensions).
LogicalResult verifyDot(Operation* op) {
  SmallVector<TensorShardingAttr> operandShardings;
  for (Value operand : op->getOperands()) {
    operandShardings.push_back(isDefiningOpExplicitReshard(operand)
                                   ? TensorShardingAttr()
                                   : getShardingBypassingBarriers(operand));
  }
  SmallVector<TensorShardingAttr> resultShardings;
  for (Value result : op->getResults()) {
    resultShardings.push_back(getShardingBypassingBarriers(result));
  }
  return verifyUnreducedAxesTransition(op, operandShardings, resultShardings,
                                       ReductionOp::SUM);
}

LogicalResult verifyStablehloCollective(Operation* op) {
  TensorShardingAttr operandSharding =
      getShardingBypassingBarriers(op->getOperand(0));
  TensorShardingAttr resultSharding =
      getShardingBypassingBarriers(op->getResult(0));
  if (!operandSharding && !resultSharding) {
    return success();
  }

  std::optional<ReductionOp> bodyRedOp =
      op->getNumRegions() > 0 ? getReductionType(op->getRegion(0))
                              : std::nullopt;

  return verifyUnreducedAxesTransition(op, {operandSharding}, {resultSharding},
                                       std::nullopt, bodyRedOp);
}

struct VerifyUnreducedAxesPass
    : public impl::VerifyUnreducedAxesPassBase<VerifyUnreducedAxesPass> {
  using VerifyUnreducedAxesPassBase::VerifyUnreducedAxesPassBase;

 protected:
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();

    auto walkResult = funcOp.walk([&](Operation* op) {
      if (isExplicitReshardOp(op)) {
        if (failed(mlir::verify(op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (auto reduceOp = dyn_cast<stablehlo::ReduceOp>(op)) {
        if (failed(verifyReduce(reduceOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (isa<stablehlo::DotGeneralOp, stablehlo::DotOp>(op)) {
        if (failed(verifyDot(op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (isa<sdy::AllReduceOp, sdy::ReduceScatterOp>(op)) {
        if (!getShardingBypassingBarriers(op->getOperand(0))) {
          op->emitOpError("operand must be sharded.");
          return WalkResult::interrupt();
        }
        if (failed(mlir::verify(op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (isa<stablehlo::AllReduceOp, stablehlo::ReduceScatterOp>(op)) {
        if (failed(verifyStablehloCollective(op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (failed(verifyCall(callOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (op->hasTrait<OpTrait::IsTerminator>()) {
        if (op->hasTrait<OpTrait::ReturnLike>()) {
          if (failed(verifyTerminator(op, funcOp))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      }

      if (failed(verifyDefaultOp(op, funcOp))) {
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
