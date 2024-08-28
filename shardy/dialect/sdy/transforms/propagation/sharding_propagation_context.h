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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_PROPAGATION_CONTEXT_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_PROPAGATION_CONTEXT_H_

#include <cassert>
#include <cstdint>
#include <functional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"

namespace mlir {
namespace sdy {

// An interface for creating the callback functions used for getting and setting
// shardings during propagation. This can also optionally enforce constraints (
// such as sharding groups) by creating appropriate callbacks and handling any
// additional logic required.
class ShardingPropagationContext {
 public:
  ShardingPropagationContext() = default;

  // Propagates tensor shardings of the given `operands` and `results` according
  // to `shardingRule`.
  //
  // NOTE: the `operands`/`results` can be any sort of ValueRange associated to
  // the Operation. For example, for CaseOp, an op with no operands, it's called
  // with the return values of each branch/region.
  LogicalResult propagateTensorShardings(
      ValueRange operands, ValueRange results,
      ArrayRef<TensorShardingAttr> operandShardings,
      ArrayRef<TensorShardingAttr> resultsShardings,
      std::function<void(TensorShardingAttr, int64_t)>
          setOperandShardingCallback,
      std::function<void(TensorShardingAttr, int64_t)>
          setResultShardingCallback,
      OpShardingRuleAttr shardingRule, PropagationDirection direction,
      const FactorPropagation& factorPropagation, bool conservativePropagation,
      Operation* op, PatternRewriter* rewriter) const;

  // Same as the overload above, except there is a single operand and result.
  LogicalResult propagateSingleTensorSharding(
      Value operand, Value result, TensorShardingAttr operandSharding,
      TensorShardingAttr resultsSharding,
      std::function<void(TensorShardingAttr)> setOperandShardingCallback,
      std::function<void(TensorShardingAttr)> setResultShardingCallback,
      OpShardingRuleAttr shardingRule, Operation* op, PatternRewriter* rewriter,
      const FactorPropagation& factorPropagation,
      PropagationDirection direction = PropagationDirection::BOTH,
      bool conservativePropagation = false) const;

  // Same as the overload above, except the operand and result shardings are
  // extracted using `getSharding` and set using `setSharding`.
  LogicalResult propagateTensorShardingsWithDefaultCallbacks(
      ValueRange operands, ValueRange results, OpShardingRuleAttr shardingRule,
      Operation* op, PatternRewriter& rewriter,
      const FactorPropagation& factorPropagation,
      PropagationDirection direction = PropagationDirection::BOTH,
      bool conservativePropagation = false) const;

  // Propagates the shardings between the operands of the `funcOp`'s terminator
  // and the `funcOp`'s result type attrs.
  LogicalResult propagateFuncResults(
      func::FuncOp funcOp, const FactorPropagation& factorPropagation) const;

  // Propagates operand/result shardings of every `FuncOp` in a `moduleOp`.
  LogicalResult propagateAllFuncResultsInModule(
      ModuleOp moduleOp, const FactorPropagation& factorPropagation) const;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_PROPAGATION_CONTEXT_H_
