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

#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"

#include <cassert>
#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_AGGRESSIVEPROPAGATIONPASS
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

namespace {

// The aggressive propagation pass that uses the default implementation of
// `AggressivePropagationPassImpl`.
struct AggressivePropagationPass
    : public impl::AggressivePropagationPassBase<AggressivePropagationPass> {
  using AggressivePropagationPassBase::AggressivePropagationPassBase;

  explicit AggressivePropagationPass(
      // NOLINTBEGIN(clang-diagnostic-shadow-field)
      bool keepShardingRules, StringRef dumpDirectory,
      bool conservativePropagation, PropagationStrategy propagationStrategy) {
    // NOLINTEND(clang-diagnostic-shadow-field)
    this->keepShardingRules = keepShardingRules;
    this->dumpDirectory = dumpDirectory.str();
    this->conservativePropagation = conservativePropagation;
    this->propagationStrategy = propagationStrategy;
  }
};

}  // namespace

LogicalResult AggressivePropagationPassImpl::propagate(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    const ShardingGroupMap& shardingGroupMap,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  SmallVector<const FactorPropagation*, 2> strategies;
  switch (propagationStrategy) {
    case PropagationStrategy::Aggressive: {
      strategies.push_back(&aggressiveFactorPropagation);
      break;
    }
    case PropagationStrategy::Basic: {
      strategies.push_back(&getBasicFactorPropagation());
      break;
    }
    case PropagationStrategy::BasicThenAggressive: {
      strategies.push_back(&getBasicFactorPropagation());
      strategies.push_back(&aggressiveFactorPropagation);
      break;
    }
  }

  for (const FactorPropagation* strategy : strategies) {
    if (failed(BasicPropagationPassImpl::propagate(moduleOp, symbolTable,
                                                   shardingGroupMap, *strategy,
                                                   getDirectionToPropagate))) {
      return failure();
    }
  }
  return success();
}

std::unique_ptr<Pass> createAggressivePropagationPass(
    bool keepShardingRules, StringRef dumpDirectory,
    bool conservativePropagation, PropagationStrategy propagationStrategy) {
  return std::make_unique<AggressivePropagationPass>(
      keepShardingRules, dumpDirectory, conservativePropagation,
      propagationStrategy);
}

}  // namespace sdy
}  // namespace mlir
