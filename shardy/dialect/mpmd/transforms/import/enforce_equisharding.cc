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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/import/sharding_constraints.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_ENFORCEEQUISHARDINGPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

// Enforces input-output equisharding constraints for MPMD functions by
// introducing TransferOps when necessary.
class EnforceEquishardingPass
    : public impl::EnforceEquishardingPassBase<EnforceEquishardingPass> {
  using EnforceEquishardingPassBase::EnforceEquishardingPassBase;

  void runOnOperation() final {
    func::FuncOp func_op = getOperation();
    if (!IsMpmdFunction(func_op) || !IsEntryPointFunction(func_op)) {
      return;
    }
    IRRewriter rewriter(func_op.getContext());
    FunctionType func_type = func_op.getFunctionType();
    auto func_ret = cast<func::ReturnOp>(func_op.front().getTerminator());

    rewriter.setInsertionPoint(func_ret);
    for (const InputOutputEquishardingConstraint& constraint : constraints) {
      Type output_mesh_type = func_type.getResult(constraint.output_index);
      Type input_mesh_type = func_type.getInput(constraint.input_index);
      if (input_mesh_type != output_mesh_type) {
        Value new_operand = rewriter.create<TransferOp>(
            func_ret->getLoc(), input_mesh_type,
            func_ret->getOperand(constraint.output_index));
        func_ret->setOperand(constraint.output_index, new_operand);
      }
    }
    UpdateFunctionType(func_op);
  }
};

}  // namespace
}  // namespace mlir::mpmd
