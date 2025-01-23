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

#include "shardy/dialect/sdy/ir/extensions/stablehlo_extensions.h"

#include <cassert>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

struct WhileShardableDataFlowOpInterface
    : public ShardableDataFlowOpInterface::ExternalModel<
          WhileShardableDataFlowOpInterface, stablehlo::WhileOp> {
  ResultRange getOpResultEdgeOwners(Operation *op) const {
    return op->getResults();
  }

  SmallVector<Value> getEdgeSources(Operation *op, Value owner) const {
    auto whileOp = cast<stablehlo::WhileOp>(op);
    auto opResult = dyn_cast<OpResult>(owner);
    assert(opResult && opResult.getOwner() == op);
    unsigned int resNum = opResult.getResultNumber();
    return {whileOp->getOperand(resNum),
            getBodyTerminatorOperands(whileOp)[resNum]};
  }

  Value getEdgeOwnerFromTarget(Operation *op, Value target) const {
    assert(getOwningOp(target) == op);
    if (auto opResult = dyn_cast<OpResult>(target)) {
      return opResult;
    }
    return cast<stablehlo::WhileOp>(op).getResult(
        cast<BlockArgument>(target).getArgNumber());
  }

  Value getEdgeOwnerFromSource(Operation *op, OpOperand &source) const {
    return op->getResult(source.getOperandNumber());
  }

  SmallVector<Value> getNonEdgeOwnerTargets(Operation *op, Value owner) const {
    auto whileOp = cast<stablehlo::WhileOp>(op);
    auto opResult = dyn_cast<OpResult>(owner);
    assert(opResult && opResult.getOwner() == op);
    unsigned int resNum = opResult.getResultNumber();
    return {whileOp.getCond().getArgument(resNum),
            whileOp.getBody().getArgument(resNum)};
  }
};

struct CaseShardableDataFlowOpInterface
    : public ShardableDataFlowOpInterface::ExternalModel<
          CaseShardableDataFlowOpInterface, stablehlo::CaseOp> {
  ResultRange getOpResultEdgeOwners(Operation *op) const {
    return op->getResults();
  }

  SmallVector<Value> getEdgeSources(Operation *op, Value owner) const {
    auto caseOp = cast<stablehlo::CaseOp>(op);
    auto opResult = dyn_cast<OpResult>(owner);
    assert(opResult && opResult.getOwner() == op);
    SmallVector<Value> sources;
    sources.reserve(caseOp.getBranches().size());
    for (Region &branch : caseOp.getBranches()) {
      sources.push_back(branch.front().getTerminator()->getOperand(
          opResult.getResultNumber()));
    }
    return sources;
  }

  Value getEdgeOwnerFromTarget(Operation *op, Value target) const {
    auto opResult = dyn_cast<OpResult>(target);
    assert(opResult && opResult.getOwner() == op);
    return opResult;
  }

  Value getEdgeOwnerFromSource(Operation *op, OpOperand &source) const {
    return op->getResult(source.getOperandNumber());
  }
};

struct OptBarrierShardableDataFlowOpInterface
    : public ShardableDataFlowOpInterface::ExternalModel<
          OptBarrierShardableDataFlowOpInterface,
          stablehlo::OptimizationBarrierOp> {
  ResultRange getOpResultEdgeOwners(Operation *op) const {
    return op->getResults();
  }

  SmallVector<Value> getEdgeSources(Operation *op, Value owner) const {
    auto opResult = dyn_cast<OpResult>(owner);
    assert(opResult && opResult.getOwner() == op);
    return {op->getOperand(opResult.getResultNumber())};
  }

  Value getEdgeOwnerFromTarget(Operation *op, Value target) const {
    auto opResult = dyn_cast<OpResult>(target);
    assert(opResult && opResult.getOwner() == op);
    return opResult;
  }

  Value getEdgeOwnerFromSource(Operation *op, OpOperand &source) const {
    return op->getResult(source.getOperandNumber());
  }
};

}  // namespace

void registerStablehloExtensions(MLIRContext *ctx) {
  // Ensure dialect is loaded before attaching interfaces.
  ctx->loadDialect<stablehlo::StablehloDialect>();
  stablehlo::WhileOp::attachInterface<WhileShardableDataFlowOpInterface>(*ctx);
  stablehlo::CaseOp::attachInterface<CaseShardableDataFlowOpInterface>(*ctx);
  stablehlo::OptimizationBarrierOp::attachInterface<
      OptBarrierShardableDataFlowOpInterface>(*ctx);
}

}  // namespace sdy
}  // namespace mlir
