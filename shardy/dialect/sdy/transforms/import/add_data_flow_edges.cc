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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_ADDDATAFLOWEDGESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

struct AddDataFlowEdgesPass
    : public impl::AddDataFlowEdgesPassBase<AddDataFlowEdgesPass> {
  using AddDataFlowEdgesPassBase::AddDataFlowEdgesPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    IRRewriter rewriter(moduleOp);

    moduleOp.walk([&](ShardableDataFlowOpInterface op) {
      // Add the data flow edges for result owners and block argument owners.
      addDataFlowEdges(op.getBlockArgumentEdgeOwners(), rewriter);
      addDataFlowEdges(op.getOpResultEdgeOwners(), rewriter);
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
