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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

void insertExplicitReshardsToTargetSharding(OpOperand* opOperand,
                                            TensorShardingAttr targetSharding,
                                            IRRewriter& rewriter,
                                            const bool insertAfterOperand) {
  Value operand = opOperand->get();
  TensorShardingAttr operandSharding = getSharding(operand);
  if (shouldReshard(operandSharding, targetSharding)) {
    if (insertAfterOperand) {
      rewriter.setInsertionPointAfterValue(operand);
    }
    auto reshardOp = rewriter.create<ReshardOp>(
        operand.getLoc(), operand,
        targetSharding
            ? targetSharding
            // Since it should reshard and `targetSharding` is empty,
            // `operandSharding` is guaranteed to be nonempty.
            : TensorShardingAttr::getFullyClosedLike(operandSharding));
    opOperand->set(reshardOp);
  }
}

void insertExplicitReshardsOnFuncReturn(Operation* op, func::FuncOp& funcOp,
                                        IRRewriter& rewriter) {
  rewriter.setInsertionPoint(op);
  for (const auto& [index, opOperand] : llvm::enumerate(op->getOpOperands())) {
    insertExplicitReshardsToTargetSharding(
        /*opOperand=*/&opOperand,
        /*targetSharding=*/getFuncResultSharding(funcOp, index), rewriter,
        /*insertAfterOperand=*/false);
  }
}

void insertExplicitReshardsOnDataFlowOp(ShardableDataFlowOpInterface& op,
                                        IRRewriter& rewriter) {
  for (Value owner : llvm::concat<Value>(op.getOpResultEdgeOwners(),
                                         op.getBlockArgumentEdgeOwners())) {
    TensorShardingAttr ownerSharding = op.transformTargetSharding(
        owner, op.getEdgeOwnerSharding(owner),
        DataFlowShardingTransformType::kBeforeEdgePropagation);
    for (OpOperand* sourceOpOperand : op.getEdgeSources(owner)) {
      insertExplicitReshardsToTargetSharding(
          /*opOperand=*/sourceOpOperand,
          /*targetSharding=*/ownerSharding, rewriter,
          /*insertAfterOperand=*/true);
    }
  }
}

struct InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {
  using InsertExplicitReshardsPassBase::InsertExplicitReshardsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    SymbolTable symbolTable(funcOp->getParentOfType<ModuleOp>());

    funcOp.walk([&](Operation* op) {
      // TODO(enver): Does not need to be part of the walk on the func, instead
      // get the terminatior with getBodyTerminator.
      if (isa<func::ReturnOp>(op)) {
        insertExplicitReshardsOnFuncReturn(op, funcOp, rewriter);
        return;
      }

      // TODO(enver): Prefer resharding the owner when multiple sources are
      // sharded in the same way.
      if (auto shardableDataFlowOp =
              dyn_cast<ShardableDataFlowOpInterface>(op)) {
        insertExplicitReshardsOnDataFlowOp(shardableDataFlowOp, rewriter);
        return;
      }

      insertExplicitReshardsOnOp(op, rewriter, symbolTable);
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
