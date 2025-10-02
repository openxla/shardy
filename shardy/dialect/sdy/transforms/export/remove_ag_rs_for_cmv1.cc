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

#include "llvm/ADT/TypeSwitch.h"
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
    IRRewriter rewriter(&getContext());

    getOperation()->walk([&](Operation* op) {
      llvm::TypeSwitch<Operation*>(op)
          .Case<sdy::AllGatherOp>([&](sdy::AllGatherOp allGatherOp) {
            if (allGatherOp.getResult().hasOneUse() &&
                mlir::isa<stablehlo::DotOp, stablehlo::DotGeneralOp>(
                    *allGatherOp.getResult().user_begin())) {
              rewriter.replaceOp(allGatherOp, allGatherOp.getTensor());
            }
          })
          .Case<sdy::ReduceScatterOp>([&](sdy::ReduceScatterOp
                                              reduceScatterOp) {
            Operation* operandDefOp =
                reduceScatterOp.getTensor().getDefiningOp();
            if (isa_and_nonnull<stablehlo::DotOp, stablehlo::DotGeneralOp>(
                    operandDefOp) &&
                getSharding(operandDefOp->getResult(0))
                    .getUnreducedAxes()
                    .empty()) {
              setShardings(operandDefOp, {reduceScatterOp.getOutSharding()});
              rewriter.replaceOp(reduceScatterOp, reduceScatterOp.getTensor());
            }
          });
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
