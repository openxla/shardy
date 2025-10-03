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
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_UNIQUIFYFUNCTIONINPUTSOUTPUTSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using ValueToReturnIndices = llvm::MapVector<Value, SmallVector<int64_t>>;


class UniquifyFunctionInputOutputsPass
    : public impl::UniquifyFunctionInputsOutputsPassBase<
          UniquifyFunctionInputOutputsPass> {
  using UniquifyFunctionInputsOutputsPassBase::
      UniquifyFunctionInputsOutputsPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      // This is not the main function. Do nothing.
      return;
    }

    Operation* return_op = func_op.getBody().front().getTerminator();
    llvm::SmallDenseSet<Value> seen_values;
    mlir::IRRewriter rewriter(&getContext());
    OpBuilder builder(&getContext());
    builder.setInsertionPoint(return_op);
    for (OpOperand& operand : return_op->getOpOperands()) {
      if (!seen_values.contains(operand.get())) {
        seen_values.insert(operand.get());
        if (!mlir::isa<BlockArgument>(operand.get())) {
          continue;
        }
      }
      auto transfer_op = TransferOp::create(builder, return_op->getLoc(), cast<MeshTensorType>(operand.get().getType()),
          operand.get());
      operand.set(transfer_op->getResult(0));
    }

    // func_op.dump();
  }
};

}  // namespace
}  // namespace mlir::mpmd
