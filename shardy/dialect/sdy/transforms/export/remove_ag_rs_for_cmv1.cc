/* Copyright 2025 The Shardy Authors.

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

#include <cassert>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_REMOVEALLGATHERREDUCESCATTERFORCMV1PASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct RemoveAllGatherReduceScatterForCMV1Pass
    : public impl::RemoveAllGatherReduceScatterForCMV1PassBase<
          RemoveAllGatherReduceScatterForCMV1Pass> {
  using RemoveAllGatherReduceScatterForCMV1PassBase::
      RemoveAllGatherReduceScatterForCMV1PassBase;

  void runOnOperation() final {
    SmallVector<Operation*> opsToReplace;

    getOperation()->walk([&](Operation* dotOp) {
      if (!mlir::isa<stablehlo::DotOp, stablehlo::DotGeneralOp>(dotOp)) {
        return;
      }

      bool hasAllGatherOperand = false;
      for (Value operand : dotOp->getOperands()) {
        if (auto allGatherOp = operand.getDefiningOp<sdy::AllGatherOp>();
            allGatherOp && allGatherOp->hasOneUse()) {
          opsToReplace.push_back(allGatherOp);
          hasAllGatherOperand = true;
        }
      }

      // If the operand is all-gather, we can keep the reduce-scatter for the
      // dot result.
      if (hasAllGatherOperand) {
        return;
      }

      if (dotOp->hasOneUse()) {
        if (auto reduceScatterOp =
                dyn_cast<sdy::ReduceScatterOp>(*dotOp->user_begin());
            reduceScatterOp &&
            getSharding(dotOp->getResult(0)).getUnreducedAxes().empty()) {
          setShardings(dotOp, {getSharding(reduceScatterOp->getResult(0))});
          opsToReplace.push_back(reduceScatterOp);
        }
      }
    });

    IRRewriter rewriter(&getContext());
    for (Operation* opToReplace : opsToReplace) {
      rewriter.replaceOp(opToReplace, opToReplace->getOperand(0));
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
