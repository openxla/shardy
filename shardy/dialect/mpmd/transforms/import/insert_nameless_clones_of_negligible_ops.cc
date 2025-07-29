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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_INSERTNAMELESSCLONEOFNEGLIBLEOPSPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

// Returns the producer of `fragment_result` if it is considered a negligible,
// i.e., it has no operands and returns a single result.
Operation* FindNegligibleProducer(OpResult fragment_result,
                                  ReturnOp fragment_return) {
  Value returned_value =
      fragment_return.getOperand(fragment_result.getResultNumber());
  if (Operation* producer = returned_value.getDefiningOp();
      producer && producer->getNumOperands() == 0 &&
      producer->getNumResults() == 1) {
    return producer;
  }
  return nullptr;
}

class InsertNamelessCloneOfNeglibleOpsPass
    : public impl::InsertNamelessCloneOfNeglibleOpsPassBase<
          InsertNamelessCloneOfNeglibleOpsPass> {
  using InsertNamelessCloneOfNeglibleOpsPassBase::
      InsertNamelessCloneOfNeglibleOpsPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) final {
    IRRewriter rewriter(func_op.getContext());
    func_op.walk([&rewriter](NamedComputationOp named_computation) {
      rewriter.setInsertionPointAfter(named_computation);
      Operation* terminator =
          named_computation.getBody()->getTerminator();
      for (OpResult result : named_computation->getResults()) {
        if (Operation* producer =
                FindNegligibleProducer(result, cast<ReturnOp>(terminator))) {
          // We only replace uses that represent ops that need mesh assignment.
          // A function's return will never be assigned to a mesh and therefore
          // should not be affected. Moreover, doing so could cause problems
          // when a named_computation produces a constant directly returned by
          // the function (e.g., as observed in an init function), as they may
          // have been explicitly assigned by the user with a named_computation,
          // i.e., such assignment must be preserved.
          rewriter.replaceUsesWithIf(
              result, Clone(rewriter, *producer, {})->getResult(0),
              [](OpOperand& use) {
                return !isa<func::ReturnOp>(use.getOwner());
              });
        }
      }
    });
  }
};

}  // namespace

}  // namespace mlir::mpmd
