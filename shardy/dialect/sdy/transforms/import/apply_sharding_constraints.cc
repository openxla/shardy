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

bool shouldApply(Value input, Operation* op) {
  if (getSharding(input)) {
    // `input` already has a sharding.
    return false;
  }

  if (input.hasOneUse()) {
    // `op` is the only use of `input`.
    return true;
  }

  if (!isa<ShardingConstraintOp>(op)) {
    // `op` is a `ManualComputationOp` and `input` has other uses.
    assert(isa<ManualComputationOp>(op));
    return false;
  }

  // `op` is dangling `ShardingConstraintOp`, and `input` has no other uses of
  // type `ShardingConstraintOp` or `ManualComputationOp`.
  return op->use_empty() &&
         llvm::none_of(input.getUsers(), [op](Operation* user) {
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
                if (shouldApply(input, shardingConstraintOp)) {
                  setSharding(input, shardingConstraintOp.getSharding());
                }
              })
          .Case<ManualComputationOp>(
              [](ManualComputationOp manualComputationOp) {
                for (auto [operand, sharding] : llvm::zip_equal(
                         manualComputationOp.getOperands(),
                         manualComputationOp.getInShardings().getShardings())) {
                  if (shouldApply(operand, manualComputationOp)) {
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
