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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATENAMEDOPSINMPMDFUNCPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

class ValidateNamedOpsInMpmdFuncPass
    : public impl::ValidateNamedOpsInMpmdFuncPassBase<
          ValidateNamedOpsInMpmdFuncPass> {
  using ValidateNamedOpsInMpmdFuncPassBase::ValidateNamedOpsInMpmdFuncPassBase;

  bool IsImmediateParentMpmdFuncOrMpmdOp(Operation* op) {
    Operation* parent_op = op->getParentOp();
    if (func::FuncOp func = dyn_cast<func::FuncOp>(parent_op)) {
      return IsMpmdFunction(func);
    }
    return sdy::inDialect<mpmd::MpmdDialect>(parent_op);
  }

  void runOnOperation() final {
    // Check named computations and named tensors are only nested in mpmd
    // function (function with topology).
    getOperation().walk([&](Operation* op) {
      if (auto named_computation = dyn_cast<NamedComputationOp>(op)) {
        if (!IsImmediateParentMpmdFuncOrMpmdOp(named_computation)) {
          emitError(named_computation->getLoc())
              << "Named computations can only be nested in mpmd functions or "
                 "mpmd ops.";
        }
      }

      if (auto named_tensor = dyn_cast<NamedTensorOp>(op)) {
        if (!IsImmediateParentMpmdFuncOrMpmdOp(named_tensor)) {
          emitError(named_tensor->getLoc())
              << "Named tensors can only be nested in mpmd functions or mpmd "
                 "ops.";
        }
      }
    });
  }
};

}  // namespace

}  // namespace mlir::mpmd

