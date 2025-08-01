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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_USER_PRIORITY_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_USER_PRIORITY_PROPAGATION_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/op_priority_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"

namespace mlir {
namespace sdy {

// The implementation class for the user-priority propagation pass.
class UserPriorityPropagationPassImpl : public OpPriorityPropagationPassImpl {
 public:
  using OpPriorityPropagationPassImpl::OpPriorityPropagationPassImpl;

 protected:
  LogicalResult propagate(
      ModuleOp moduleOp, const SymbolTable& symbolTable,
      const ShardingGroupMap& shardingGroupMap,
      GetDirectionToPropagateFn getDirectionToPropagate) override;

  // Current module dump index.
  int dumpIndex = 0;
};

// Runs the user-priority propagation algorithm (see
// `UserPriorityPropagationPass`).
std::unique_ptr<Pass> createUserPriorityPropagationPass(
    const PropagationOptions& options, int dumpIndex = 0);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_USER_PRIORITY_PROPAGATION_H_
