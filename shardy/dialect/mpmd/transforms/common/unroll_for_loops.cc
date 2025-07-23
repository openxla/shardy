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

#include <cstdint>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_UNROLLFORLOOPSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

void SetUnrollCounter(Operation* op, int counter, RewriterBase& rewriter) {
  op->setAttr(kUnrollCounterAttrName, rewriter.getUI32IntegerAttr(counter));
}

Value GetUint32Constant(RewriterBase& rewriter, Location loc, uint32_t value) {
  auto shape = RankedTensorType::get(
      {}, rewriter.getIntegerType(32, /*isSigned=*/false));
  return rewriter.create<stablehlo::ConstantOp>(
      loc, DenseIntElementsAttr::get(shape, value));
}

// Requires: the unroll factor to be equal to the number of iterations.
void FullyUnrollForOp(ForOp for_op, RewriterBase& rewriter) {
  const uint32_t num_iterations = for_op.getIterations();

  Block* for_body = for_op.getBody();
  rewriter.setInsertionPoint(for_op);

  for (uint32_t unroll_counter = 0; unroll_counter < num_iterations;
       ++unroll_counter) {
    // Create the unrolled index constant.
    Value unrolled_index =
        GetUint32Constant(rewriter, for_op.getLoc(), unroll_counter);
    SetUnrollCounter(unrolled_index.getDefiningOp(), unroll_counter, rewriter);

    IRMapping block_args_to_for_op_operands;
    // The unrolled index argument is mapped to the newly created index
    // constant.
    block_args_to_for_op_operands.map(for_body->getArguments().back(),
                                      unrolled_index);
    // Map all arguments of the block to the respective for operands.
    for (auto [operand, arg] :
        llvm::zip(for_op.getOperands(), for_body->getArguments().drop_back())) {
      block_args_to_for_op_operands.map(arg, operand);
    }

    for (Operation& for_body_op : for_body->getOperations()) {
      if (for_body_op.hasTrait<OpTrait::IsTerminator>()) {
        // The new operands of the for-loop are the results of the last
        // unrolled iteration.
        std::vector<Value> new_operands;
        for (Value operand : for_body_op.getOperands()) {
          // We have no free-variables in the for-loop, so all operands are
          // block arguments.
          new_operands.push_back(
              block_args_to_for_op_operands.lookup(operand));
        }
        if (unroll_counter == num_iterations - 1) {
          // In the last iteration of the loop, the users of the `for` loop are
          // other ops.
          rewriter.replaceAllOpUsesWith(for_op, new_operands);
        } else {
          // In any other iteration of the loop, the uses of the `for` loop are
          // the `for` loop itself.
          for_op->setOperands(new_operands);
        }
      } else {
        Operation* unrolled_op =
            rewriter.clone(for_body_op, block_args_to_for_op_operands);
        // We annotate all unrolled ops with the unroll counter, so that we have
        // enough information to generate the pipeline schedule.
        SetUnrollCounter(unrolled_op, unroll_counter, rewriter);
      }
    }
  }
  rewriter.eraseOp(for_op);
}

class UnrollForLoopsPass
    : public impl::UnrollForLoopsPassBase<UnrollForLoopsPass> {
  using UnrollForLoopsPassBase::UnrollForLoopsPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());
    auto walk_result = func_op.getBody().walk([&rewriter](ForOp op) {
      // Crashes if pre-condition is not met.
      SDY_CHECK(op.getIterations() == op.getUnrollFactor())
          << "The unroll factor is required to be the same as the number of "
          << "iterations.";
      // TODO: b/372460554 - Support nested mpmd.for loops.
      // NOTE: At the moment, users will be able to nest for loops using
      // mpmd.calls as an indirection, this could create odd pipeline schedules,
      // and needs to be addressed (either disallow pipeline scheduling on
      // nested loops, or change the scheduler to consider more than one
      // call/unroll_counter).
      if (op->getParentOfType<ForOp>()) {
        op->emitError(
            "Nested fori loops aren't supported. Please contact OWNERs if you ")
            << "need this feature.";
        return WalkResult::interrupt();
      }
      FullyUnrollForOp(op, rewriter);
      // No nested for loops, so no need to visit the body.
      return WalkResult::skip();
    });
    if (walk_result.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
