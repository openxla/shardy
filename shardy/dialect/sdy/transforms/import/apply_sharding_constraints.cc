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

bool shouldApply(Value input, ShardingConstraintOp shardingConstraintOp) {
  if (getSharding(input)) {
    // `input` already has a sharding.
    return false;
  }

  if (input.hasOneUse()) {
    // `shardingConstraintOp` is the only use of `input`.
    return true;
  }

  // `shardingConstraintOp` is dangling and `input` has no other uses of type
  // `ShardingConstraintOp`.
  return shardingConstraintOp.use_empty() &&
         llvm::none_of(input.getUsers(), [&](Operation* user) {
           return user != shardingConstraintOp &&
                  isa<ShardingConstraintOp>(user);
         });
}

struct ApplyShardingConstraintsPass
    : public impl::ApplyShardingConstraintsPassBase<
          ApplyShardingConstraintsPass> {
  using ApplyShardingConstraintsPassBase::ApplyShardingConstraintsPassBase;

  void runOnOperation() final {
    getOperation().walk([](ShardingConstraintOp shardingConstraintOp) {
      Value input = shardingConstraintOp.getInput();
      if (shouldApply(input, shardingConstraintOp)) {
        setSharding(input, shardingConstraintOp.getSharding());
      }
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
