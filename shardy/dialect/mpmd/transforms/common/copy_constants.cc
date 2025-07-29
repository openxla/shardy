/* Copyright 2025 The MPMD Authors.

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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_COPYCONSTANTSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

// Finds the HLO producer of `mesh_tensor`. I.e., given a tensor that lives at
// the top level of the function (e.g., produced/consumed by a fragment), find
// its producer, which is nested in a fragment.
// TODO: jupvfranco - consider adapting this code to go through control-flow
// ops.
Operation* FindHloProducer(Value mesh_tensor) {
  auto op_result = dyn_cast_or_null<OpResult>(mesh_tensor);
  if (!op_result) {
    return nullptr;
  }
  Operation* mpmd_producer = op_result.getOwner();

  if (auto transfer = dyn_cast_or_null<TransferOp>(mpmd_producer)) {
    return FindHloProducer(transfer.getOperand());
  }

  if (auto fragment = dyn_cast_or_null<FragmentOp>(mpmd_producer)) {
    return fragment.getBody()
        ->getTerminator()
        ->getOperand(op_result.getResultNumber())
        .getDefiningOp();
  }

  return nullptr;
}

class CopyConstantsPass
    : public impl::CopyConstantsPassBase<CopyConstantsPass> {
  using CopyConstantsPassBase::CopyConstantsPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());
    Block& block = func_op.getBody().front();
    for (FragmentOp fragment : llvm::reverse(block.getOps<FragmentOp>())) {
      for (OpOperand& operand : fragment->getOpOperands()) {
        if (auto hlo_producer = FindHloProducer(operand.get());
            hlo_producer && hlo_producer->hasTrait<OpTrait::ConstantLike>()) {
          rewriter.setInsertionPoint(fragment.getBody(),
                                     fragment.getBody()->begin());
          rewriter.replaceAllUsesWith(
              fragment.getBody()->getArgument(operand.getOperandNumber()),
              rewriter.clone(*hlo_producer)->getResult(0));
        }
      }
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
