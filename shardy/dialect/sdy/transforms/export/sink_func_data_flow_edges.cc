/* Copyright 2024 The Shardy Authors.

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
#include <memory>  // IWYU pragma: keep
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/debugging/source_sharding.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SINKFUNCDATAFLOWEDGESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct SinkFuncDataFlowEdgesPass
    : public impl::SinkFuncDataFlowEdgesPassBase<SinkFuncDataFlowEdgesPass> {
  using SinkFuncDataFlowEdgesPassBase::SinkFuncDataFlowEdgesPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    // Copy the sharding from data flow edges to the data flow ops.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
      // Since we are doing the walk in preorder with a forward iterator, ops
      // are walked before their users and regions. Since `DataFlowEdgeOp` can
      // only appear inside the data flow op's region or as its user, we always
      // encounter the data flow op before their data flow edges. This means it
      // is safe to erase the `FuncDataFlowEdgeOp` at this point. We need the
      // skip at the end because it's a condition to erase the op. See the
      // documentation for `Operation::walk` for more details.
      if (isa<FuncDataFlowEdgeOp>(op)) {
        FuncDataFlowEdgeOp funcEdgeOp = cast<FuncDataFlowEdgeOp>(op);
        Value operand = funcEdgeOp.getOperand();
        Value result = funcEdgeOp.getResult();
        TensorShardingAttr operandSharding = getSharding(operand);
        if (TensorShardingAttr sharding = getSharding(result)) {
          setSharding(operand, sharding);
        } else if (operandSharding) {
          setSharding(operand,
                      TensorShardingAttr::getFullyOpenLike(operandSharding));
        }
        rewriter.replaceOp(funcEdgeOp, operand);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
