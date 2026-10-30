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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "shardy/dialect/sdy/transforms/export/utils.h"
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

// Verifies that `sourceSharding` and `targetSharding` have identical unreduced
// axes and, when non-empty, identical `ReductionOp`.
LogicalResult verifyUnreducedShardingEqual(TensorShardingAttr sourceSharding,
                                           TensorShardingAttr targetSharding,
                                           Operation* op,
                                           const llvm::Twine& errorMsgSuffix) {
  ArrayRef<AxisRefAttr> sourceAxes = sourceSharding
                                         ? sourceSharding.getUnreducedAxes()
                                         : ArrayRef<AxisRefAttr>{};
  ArrayRef<AxisRefAttr> targetAxes = targetSharding
                                         ? targetSharding.getUnreducedAxes()
                                         : ArrayRef<AxisRefAttr>{};
  if (failed(verifyEqual(sourceAxes, targetAxes, op, errorMsgSuffix))) {
    return failure();
  }
  if (!sourceAxes.empty() &&
      sourceSharding.getReductionOp() != targetSharding.getReductionOp()) {
    return op->emitOpError(
               "cannot change the reduction operator of kept unreduced axes "
               "from ")
           << stringifyReductionOp(sourceSharding.getReductionOp()) << " to "
           << stringifyReductionOp(targetSharding.getReductionOp()) << ".";
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
    TensorShardingAttr operandSharding = getSharding(operand);
    TensorShardingAttr argSharding = getSharding(callee.getArgument(i));
    if (failed(verifyUnreducedShardingEqual(
            operandSharding, argSharding, callOp,
            llvm::Twine("' at call argument ") + llvm::Twine(i) + "."))) {
      return failure();
    }
  }

  // Verify callee results against call results.
  for (int64_t i = 0; i < callOp.getNumResults(); ++i) {
    TensorShardingAttr funcResultSharding = getFuncResultSharding(callee, i);
    TensorShardingAttr resultSharding = getSharding(callOp.getResult(i));
    if (failed(verifyUnreducedShardingEqual(
            funcResultSharding, resultSharding, callOp,
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
  return verifyUnreducedShardingEqual(
      getSharding(operand), resultSharding, op,
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
      if (!resultSharding) {
        continue;
      }
      if (failed(verifyReturnPair(op->getOperand(i), resultSharding, op, i))) {
        return failure();
      }
    }
    return success();
  }

  // Verify nested terminator: the return operands must match the parent op
  // results (except for WhileOp's cond region, which returns a boolean
  // predicate).
  Operation* parentOp = op->getParentOp();
  if (auto whileOp = dyn_cast<stablehlo::WhileOp>(parentOp)) {
    if (op->getParentRegion() == &whileOp.getCond()) {
      SmallVector<AxisRefAttr> condAxes;
      appendUnreducedAxes(getSharding(op->getOperand(0)), condAxes);
      return verifyEqual(
          condAxes, {}, op,
          "' at while condition return value 0 without a blessed operation.");
    }
  }
  if (parentOp->getNumResults() > 0) {
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      if (i < parentOp->getNumResults()) {
        TensorShardingAttr resultSharding = getSharding(parentOp->getResult(i));
        if (auto manualCompOp = dyn_cast<sdy::ManualComputationOp>(parentOp)) {
          if (resultSharding) {
            resultSharding =
                eraseManualAxes(resultSharding, manualCompOp.getManualAxes(),
                                resultSharding.getMesh(manualCompOp));
          }
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

LogicalResult verifyWhileOp(stablehlo::WhileOp whileOp) {
  for (int64_t i = 0; i < whileOp->getNumOperands(); ++i) {
    Value operand = whileOp->getOperand(i);
    if (isDefiningOpExplicitReshard(operand)) {
      continue;
    }
    TensorShardingAttr operandSharding = getSharding(operand);
    TensorShardingAttr resultSharding = getSharding(whileOp->getResult(i));
    if (failed(verifyUnreducedAxesTransition(
            whileOp, {operandSharding}, {resultSharding},
            /*expectedIntroducedRedOp=*/std::nullopt))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyRegionBranchOp(Operation* op) {
  // Region branch ops (CaseOp, IfOp) only take a predicate/index operand;
  // their results are verified against each branch terminator by
  // verifyTerminator.
  SmallVector<TensorShardingAttr> operandShardings;
  for (Value operand : op->getOperands()) {
    operandShardings.push_back(isDefiningOpExplicitReshard(operand)
                                   ? TensorShardingAttr()
                                   : getSharding(operand));
  }
  return verifyUnreducedAxesTransition(
      op, operandShardings, {},
      /*expectedIntroducedRedOp=*/std::nullopt);
}

TensorShardingAttr retainOnlyManualUnreducedAxes(
    TensorShardingAttr sharding, ArrayRef<AxisRefAttr> manualAxes,
    Operation* op) {
  if (!sharding || sharding.getUnreducedAxes().empty()) {
    return TensorShardingAttr();
  }
  SmallVector<AxisRefAttr> manualUnreduced;
  for (AxisRefAttr axis : sharding.getUnreducedAxes()) {
    for (AxisRefAttr manualAxis : manualAxes) {
      if (std::optional<AxisRefAttr> overlap = axis.getOverlap(manualAxis)) {
        manualUnreduced.push_back(*overlap);
      }
    }
  }
  if (MeshAttr mesh = sharding.getMesh(op)) {
    sortAndMergeAxes(manualUnreduced, mesh);
  }
  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshOrRef(),
                                 sharding.getDimShardings(),
                                 sharding.getReplicatedAxes(), manualUnreduced,
                                 sharding.getReductionOp());
}

LogicalResult verifyManualComputationOp(
    sdy::ManualComputationOp manualComputationOp) {
  ArrayRef<AxisRefAttr> manualAxes = manualComputationOp.getManualAxes();
  // Verify non-manual unreduced axes across the operand -> body argument
  // boundary (non-manual outputs are verified by verifyTerminator).
  for (int64_t i = 0; i < manualComputationOp.getNumOperands(); ++i) {
    Value operand = manualComputationOp.getOperand(i);
    if (isDefiningOpExplicitReshard(operand)) {
      continue;
    }
    TensorShardingAttr operandSharding = getSharding(operand);
    TensorShardingAttr operandNonManualSharding =
        operandSharding
            ? eraseManualAxes(operandSharding, manualAxes,
                              operandSharding.getMesh(manualComputationOp))
            : TensorShardingAttr();
    TensorShardingAttr argSharding =
        getSharding(manualComputationOp.getBody().getArgument(i));
    if (failed(verifyUnreducedShardingEqual(
            operandNonManualSharding, argSharding, manualComputationOp,
            llvm::Twine("' at argument ") + llvm::Twine(i) + "."))) {
      return failure();
    }
  }

  // Group manual unreduced axes by `ReductionOp` across operands and results so
  // multi-result ops with unaffected results or multiple reduction ops verify
  // each reduction kind independently.
  llvm::SmallDenseMap<ReductionOp, SmallVector<TensorShardingAttr>>
      operandShardingsByOp;
  llvm::SmallDenseMap<ReductionOp, SmallVector<TensorShardingAttr>>
      resultShardingsByOp;
  SmallVector<ReductionOp> seenReductionOps;
  auto recordSharding =
      [&](TensorShardingAttr sharding,
          llvm::SmallDenseMap<ReductionOp, SmallVector<TensorShardingAttr>>&
              map) {
        if (!sharding || sharding.getUnreducedAxes().empty()) {
          return;
        }
        ReductionOp redOp = sharding.getReductionOp();
        if (!llvm::is_contained(seenReductionOps, redOp)) {
          seenReductionOps.push_back(redOp);
        }
        map[redOp].push_back(sharding);
      };

  for (Value operand : manualComputationOp.getOperands()) {
    if (!isDefiningOpExplicitReshard(operand)) {
      recordSharding(retainOnlyManualUnreducedAxes(
                         getSharding(operand), manualAxes, manualComputationOp),
                     operandShardingsByOp);
    }
  }
  for (Value result : manualComputationOp.getResults()) {
    recordSharding(retainOnlyManualUnreducedAxes(
                       getSharding(result), manualAxes, manualComputationOp),
                   resultShardingsByOp);
  }

  for (ReductionOp redOp : seenReductionOps) {
    if (failed(verifyUnreducedAxesTransition(
            manualComputationOp, operandShardingsByOp[redOp],
            resultShardingsByOp[redOp],
            /*expectedIntroducedRedOp=*/redOp))) {
      return failure();
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
                                   : getSharding(operand));
  }
  SmallVector<TensorShardingAttr> resultShardings;
  resultShardings.reserve(op->getNumResults());
  std::optional<ReductionOp> expectedIntroducedRedOp;
  for (Value result : op->getResults()) {
    TensorShardingAttr resultSharding = getSharding(result);
    resultShardings.push_back(resultSharding);
    if (resultSharding && !resultSharding.getUnreducedAxes().empty()) {
      if (expectedIntroducedRedOp &&
          *expectedIntroducedRedOp != resultSharding.getReductionOp()) {
        return op->emitOpError(
            "results introduce unreduced axes with conflicting reduction ops");
      }
      expectedIntroducedRedOp = resultSharding.getReductionOp();
    }
  }
  return verifyUnreducedAxesTransition(
      op, operandShardings, resultShardings,
      /*expectedIntroducedRedOp=*/expectedIntroducedRedOp);
}

LogicalResult verifyReduce(stablehlo::ReduceOp reduceOp) {
  SmallVector<TensorShardingAttr> operandShardings;
  for (Value operand : reduceOp.getInputs()) {
    operandShardings.push_back(isDefiningOpExplicitReshard(operand)
                                   ? TensorShardingAttr()
                                   : getSharding(operand));
  }
  SmallVector<TensorShardingAttr> resultShardings;
  for (Value result : reduceOp.getResults()) {
    resultShardings.push_back(getSharding(result));
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
                                   : getSharding(operand));
  }
  SmallVector<TensorShardingAttr> resultShardings;
  for (Value result : op->getResults()) {
    resultShardings.push_back(getSharding(result));
  }
  return verifyUnreducedAxesTransition(op, operandShardings, resultShardings,
                                       ReductionOp::SUM);
}

LogicalResult verifyStablehloCollective(Operation* op) {
  TensorShardingAttr operandSharding = getSharding(op->getOperand(0));
  TensorShardingAttr resultSharding = getSharding(op->getResult(0));
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
        if (!getSharding(op->getOperand(0))) {
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

      if (auto whileOp = dyn_cast<stablehlo::WhileOp>(op)) {
        if (failed(verifyWhileOp(whileOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (isa<stablehlo::CaseOp, stablehlo::IfOp>(op)) {
        if (failed(verifyRegionBranchOp(op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (auto manualCompOp = dyn_cast<sdy::ManualComputationOp>(op)) {
        if (failed(verifyManualComputationOp(manualCompOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      if (op->hasTrait<OpTrait::IsTerminator>()) {
        if (isa<func::FuncOp, stablehlo::WhileOp, stablehlo::CaseOp,
                stablehlo::IfOp, sdy::ManualComputationOp>(op->getParentOp())) {
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
