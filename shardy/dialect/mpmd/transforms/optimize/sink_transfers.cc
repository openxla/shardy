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
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_SINKTRANSFERSPASS
#include "shardy/dialect/mpmd/transforms/optimize/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

class SinkTransfersPass
    : public impl::SinkTransfersPassBase<SinkTransfersPass> {
  using SinkTransfersPassBase::SinkTransfersPassBase;

 protected:
  void runOnFunc(FuncOp func) override {
    IRRewriter rewriter(func.getContext());
    SmallVector<TransferOp> transfers;
    func.walk([&](TransferOp transfer) { transfers.push_back(transfer); });

    // Process in reverse program order so that consumers are sunk before
    // their producers, effectively dragging the entire chain downward.
    for (TransferOp transfer : llvm::reverse(transfers)) {
      if (transfer.use_empty()) {
        continue;
      }

      Operation* earliest_user = nullptr;
      for (Operation* user : transfer->getUsers()) {
        Operation* ancestor = GetAncestorInBlock(transfer->getBlock(), user);
        if (!ancestor) {
          continue;
        }
        if (!earliest_user || ancestor->isBeforeInBlock(earliest_user)) {
          earliest_user = ancestor;
        }
      }

      if (earliest_user && transfer->isBeforeInBlock(earliest_user)) {
        rewriter.moveOpBefore(transfer, earliest_user);
      }
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
