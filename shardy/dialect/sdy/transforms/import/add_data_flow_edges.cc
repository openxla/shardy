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

#include <memory>  // IWYU pragma: keep

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_ADDDATAFLOWEDGESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

struct AddDataFlowEdgesPass
    : public impl::AddDataFlowEdgesPassBase<AddDataFlowEdgesPass> {
  using AddDataFlowEdgesPassBase::AddDataFlowEdgesPassBase;

  void addDataFlowEdges(ValueRange edgeOwners, IRRewriter& rewriter) {
    // We are iterating the owners in a reversed order because we set the
    // insertion point after each value and we would like to keep the data flow
    // edges for the arguments/results in the same order as they appear.
    for (Value edgeOwner : llvm::reverse(edgeOwners)) {
      rewriter.setInsertionPointAfterValue(edgeOwner);
      if (!isStaticShapedType(edgeOwner.getType())) {
        // Skip non-static-shaped tensors, e.g., tokens.
        continue;
      }
      auto dataFlowEdge = rewriter.create<DataFlowEdgeOp>(
          edgeOwner.getLoc(), edgeOwner, getSharding(edgeOwner));
      rewriter.replaceAllUsesExcept(edgeOwner, dataFlowEdge, dataFlowEdge);
    }
  }

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);

    funcOp.walk([&](ShardableDataFlowOpInterface op) {
      // Add the data flow edges for result owners and block argument owners.
      addDataFlowEdges(op.getBlockArgumentEdgeOwners(), rewriter);
      addDataFlowEdges(op.getOpResultEdgeOwners(), rewriter);
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
