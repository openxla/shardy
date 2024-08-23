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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_APPLYSHARDINGCONSTRAINTSPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

bool shouldApply(Value input, TensorShardingAttr sharding, Operation* op) {
  if (getSharding(input) || input.getDefiningOp<DataFlowEdgeOp>()) {
    // `input` already has a sharding or is produced by a `DataFlowEdgeOp`.
    return false;
  }

  // This condition isn't fundamentally needed, but is here for parity with
  // GSPMD, we should revisit in the future.
  // TODO(b/358627707): revisit this condition.
  if (!sharding.isFullyClosed()) {
    // `sharding` has an open dimension.
    return false;
  }

  // TODO(b/358627707): revisit restricting to a single use if not dangling.
  // Return true if `input` has no other uses of type `ShardingConstraintOp` or
  // `ManualComputationOp`
  return llvm::none_of(input.getUsers(), [op](Operation* user) {
           return user != op &&
                  isa<ShardingConstraintOp, ManualComputationOp>(user);
         });
}

struct ApplyShardingConstraintsPass
    : public impl::ApplyShardingConstraintsPassBase<
          ApplyShardingConstraintsPass> {
  using ApplyShardingConstraintsPassBase::ApplyShardingConstraintsPassBase;

  void runOnOperation() final {
    getOperation().walk([](Operation* op) {
      TypeSwitch<Operation*>(op)
          .Case<ShardingConstraintOp>(
              [](ShardingConstraintOp shardingConstraintOp) {
                Value input = shardingConstraintOp.getInput();
                TensorShardingAttr sharding =
                    shardingConstraintOp.getSharding();
                if (shouldApply(input, sharding, shardingConstraintOp)) {
                  setSharding(input, sharding);
                }
              })
          .Case<ManualComputationOp>(
              [](ManualComputationOp manualComputationOp) {
                for (auto [operand, sharding] : llvm::zip_equal(
                         manualComputationOp.getOperands(),
                         manualComputationOp.getInShardings().getShardings())) {
                  if (shouldApply(operand, sharding, manualComputationOp)) {
                    setSharding(operand, sharding);
                  }
                }
              });
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
