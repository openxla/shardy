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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_SHARDING_WALKER_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_SHARDING_WALKER_H_

#include <cstdint>
#include <functional>
#include <variant>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Reference to a specific result of a function.
struct FuncResult {
  func::FuncOp funcOp;
  int64_t resNum;

  FuncResult(func::FuncOp funcOp, int64_t resNum)
      : funcOp(funcOp), resNum(resNum) {}
};
// A `Value` or a reference to a function result. Should be passed by const
// reference, as `sizeof(ValueOrFuncResult) >= 16`.
using ValueOrFuncResult = std::variant<Value, FuncResult>;

using ConsumeOpFn = std::function<void(Operation*)>;

using ConsumeShardingFn = std::function<void(TensorShardingAttr)>;
using ConsumeShardingAndTensorFn =
    std::function<void(TensorShardingAttr, const ValueOrFuncResult&)>;

using TransformShardingFn =
    std::function<TensorShardingAttr(TensorShardingAttr)>;
using TransformShardingForTensorFn = std::function<TensorShardingAttr(
    TensorShardingAttr, const ValueOrFuncResult&)>;

// Updates the sharding of `valueOrFuncResult` by applying `transformFn` on it.
void transformSharding(const ValueOrFuncResult& valueOrFuncResult,
                       TransformShardingFn transformFn);

// Walks the given `rootOp` in forward pre-order and applies `consumeFn` on
// any `TensorShardingAttr` encountered and corresponding `ValueOrFuncResult`.
//
// In addition, applies `consumeOpFn` on every encountered op, before consuming
// its shardings.
void walkShardings(
    Operation* rootOp, ConsumeShardingAndTensorFn consumeFn,
    ConsumeOpFn consumeOpFn = [](Operation*) {});

// Walks the given `rootOp` in forward pre-order and applies `consumeFn` on
// any `TensorShardingAttr` encountered.
//
// In addition, applies `consumeOpFn` on every encountered op, before consuming
// its shardings.
void walkShardings(
    Operation* rootOp, ConsumeShardingFn consumeFn,
    ConsumeOpFn consumeOpFn = [](Operation*) {});

// Walks the given `rootOp` in forward pre-order and updates any
// `TensorShardingAttr` encountered by applying `transformFn` on it and the
// corresponding `ValueOrFuncResult`.
//
// In addition, applies `consumeOpFn` on every encountered op, before
// transforming its shardings.
void transformShardings(
    Operation* rootOp, TransformShardingForTensorFn transformFn,
    ConsumeOpFn consumeOpFn = [](Operation*) {});

// Walks the given `rootOp` in forward pre-order and updates any
// `TensorShardingAttr` encountered by applying `transformFn` on it.
//
// In addition, applies `consumeOpFn` on every encountered op, before
// transforming its shardings.
void transformShardings(
    Operation* rootOp, TransformShardingFn transformFn,
    ConsumeOpFn consumeOpFn = [](Operation*) {});

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_SHARDING_WALKER_H_
