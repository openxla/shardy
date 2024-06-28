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

#include "shardy/dialect/sdy/ir/data_flow_utils.h"

#include <cassert>
#include <functional>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

bool isDataFlowOp(Operation* op) {
  return isa<stablehlo::CaseOp, stablehlo::OptimizationBarrierOp,
             stablehlo::WhileOp>(op);
}

Value getDataFlowEdgeRoot(Value target) {
  if (auto opResult = dyn_cast<OpResult>(target);
      opResult && isDataFlowOp(opResult.getOwner())) {
    return opResult;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(target);
      blockArg && isDataFlowOp(blockArg.getOwner()->getParentOp())) {
    return blockArg.getOwner()->getParentOp()->getResult(
        blockArg.getArgNumber());
  }
  return nullptr;
}

Value getDataFlowEdgeRoot(OpOperand& source) {
  Operation* op = source.getOwner();
  if (isDataFlowOp(op)) {
    return op->getResult(source.getOperandNumber());
  }

  if (isa<stablehlo::ReturnOp>(op) && isDataFlowOp(op->getParentOp())) {
    return op->getParentOp()->getResult(source.getOperandNumber());
  }

  return nullptr;
}

}  // namespace

ValueRange getDataFlowEdgeRoots(Operation* op) {
  if (isDataFlowOp(op)) {
    return op->getResults();
  }
  return ValueRange();
}

DataFlowEdgeOp getDataFlowEdge(Value target) {
  return DataFlowEdgeOp::getDataFlowEdgeUser(getDataFlowEdgeRoot(target));
}

DataFlowEdgeOp getDataFlowEdge(OpOperand& source) {
  return DataFlowEdgeOp::getDataFlowEdgeUser(getDataFlowEdgeRoot(source));
}

SmallVector<Value> getDataFlowSources(DataFlowEdgeOp dataFlowEdge) {
  auto opResult = dyn_cast<OpResult>(dataFlowEdge.getInput());
  assert(opResult && isDataFlowOp(opResult.getOwner()));
  int resNum = opResult.getResultNumber();
  return TypeSwitch<Operation*, SmallVector<Value>>(opResult.getOwner())
      .Case<stablehlo::CaseOp>([&](stablehlo::CaseOp caseOp) {
        SmallVector<Value> sources;
        sources.reserve(caseOp.getBranches().size());
        for (Region& branch : caseOp.getBranches()) {
          sources.push_back(branch.front().getTerminator()->getOperand(resNum));
        }
        return sources;
      })
      .Case<stablehlo::OptimizationBarrierOp>(
          [&](stablehlo::OptimizationBarrierOp optBarrierOp)
              -> SmallVector<Value> {
            return {optBarrierOp->getOperand(resNum)};
          })
      .Case<stablehlo::WhileOp>(
          [&](stablehlo::WhileOp whileOp) -> SmallVector<Value> {
            return {whileOp->getOperand(resNum),
                    getBodyTerminatorOperands(whileOp)[resNum]};
          });
}

void forEachNonRootDataFlowTarget(DataFlowEdgeOp dataFlowEdge,
                                std::function<void(Value)> fn) {
  auto opResult = dyn_cast<OpResult>(dataFlowEdge.getInput());
  assert(opResult && isDataFlowOp(opResult.getOwner()));
  if (auto whileOp = dyn_cast<stablehlo::WhileOp>(opResult.getOwner())) {
    int resNum = opResult.getResultNumber();
    fn(whileOp.getCond().getArgument(resNum));
    fn(whileOp.getBody().getArgument(resNum));
  }
}

}  // namespace sdy
}  // namespace mlir
