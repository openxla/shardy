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

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_DELAYINFERREDFRAGMENTSPASS
#define GEN_PASS_DEF_DELAYTRANSFERSFROMCPUPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

Operation* FindFirstUserInProgramOrder(Operation* op) {
  if (op->use_empty()) {
    return nullptr;
  }
  return *llvm::min_element(op->getUsers(), [](Operation* lhs, Operation* rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
}

class DelayInferredFragmentsPass
    : public impl::DelayInferredFragmentsPassBase<DelayInferredFragmentsPass> {
  using DelayInferredFragmentsPassBase::DelayInferredFragmentsPassBase;

 private:
  void runOnFunc(func::FuncOp main_func) override {
    if (!IsMpmdFunction(main_func) || !IsEntryPointFunction(main_func)) {
      return;
    }

    IRRewriter rewriter(main_func.getContext());
    Block& block = main_func.getBody().front();
    // Iterate over all ops in reverse order, so that we guarantee we don't
    // visit the same inferred fragment twice. Also to make sure we handle
    // chains of inferred fragments correctly.
    for (Operation& operation :
         llvm::make_early_inc_range(llvm::reverse(block.getOperations()))) {
      if (auto fragment = dyn_cast<FragmentOp>(&operation)) {
        if (fragment.isUserFragment()) {
          continue;
        }

        if (Operation* first_user = FindFirstUserInProgramOrder(fragment)) {
          rewriter.moveOpBefore(fragment, first_user);
        }
      }
    }
  }
};

bool IsOnCpuMesh(Value value) {
  return cast<MeshTensorType>(value.getType())
      .getMeshName()
      .ends_with(kCpuMeshSuffix);
}


class DelayTransfersFromCpuPass
    : public impl::DelayTransfersFromCpuPassBase<DelayTransfersFromCpuPass> {
  using DelayTransfersFromCpuPassBase::DelayTransfersFromCpuPassBase;

 private:
  void runOnFunc(func::FuncOp main_func) override {
    if (!IsMpmdFunction(main_func) || !IsEntryPointFunction(main_func)) {
      return;
    }

    IRRewriter rewriter(main_func.getContext());
    Block& block = main_func.getBody().front();

    // Copy all transfers to a vector so that rewriting the module
    // doesn't affect the iteration order.
    std::vector<TransferOp> transfers(block.getOps<TransferOp>().begin(),
                                      block.getOps<TransferOp>().end());
    for (auto transfer : transfers) {
      auto operand_type = cast<MeshTensorType>(transfer.getOperand().getType());
      if (IsOnCpuMesh(transfer.getOperand()) ||
      operand_type.isOnHost()) {
        if (auto first_user = FindFirstUserInProgramOrder(transfer)) {
          rewriter.moveOpBefore(transfer, first_user);
        }
      }
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
