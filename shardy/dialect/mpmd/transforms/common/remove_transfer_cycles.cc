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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_REMOVETRANSFERCYCLESPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

// Walks the chain of device-only transfers leading to `transfer_op` to find an
// operand with the same type. If so, this is a root of the cycle. Note that
// this function finds the first such root.
//
// Returns nullptr if no such root is found.
Value FindDeviceOnlyTransferCycleClosestRoot(TransferOp transfer_op) {
  TransferOp transfer_parent = transfer_op;
  while (transfer_parent) {
    auto parent_operand_type = transfer_parent.getTensor().getType();
    // TODO: b/397933351 - This doesn't handle memory kind attributes. We likely
    // don't want to use attributes for the memory kinds on transfers, but if we
    // do, then we should handle them here.
    if (parent_operand_type.getMemoryKind() &&
        parent_operand_type.getMemoryKind().getValue() != kMemoryKindDevice) {
      return nullptr;
    }
    if (parent_operand_type == transfer_op.getType()) {
      return transfer_parent.getTensor();
    }
    transfer_parent = dyn_cast_if_present<TransferOp>(
        transfer_parent.getTensor().getDefiningOp());
  }

  return nullptr;
}

class RemoveTransferCyclesPass
    : public impl::RemoveTransferCyclesPassBase<RemoveTransferCyclesPass> {
  using RemoveTransferCyclesPassBase::RemoveTransferCyclesPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(&getContext());
    // Walk the func in reverse, so that we can delete user
    // ops without having to worry about invalidating iterators.
    for (Operation& op : llvm::make_early_inc_range(
             llvm::reverse(func_op.front().getOperations()))) {
      if (auto transfer_op = dyn_cast<TransferOp>(&op)) {
        // Note that because we walk in reverse, it suffices to keep replacing
        // the closest root we find.
        if (Value root = FindDeviceOnlyTransferCycleClosestRoot(transfer_op)) {
          rewriter.replaceAllUsesWith(transfer_op, root);
          rewriter.eraseOp(transfer_op);
        }
      }
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
