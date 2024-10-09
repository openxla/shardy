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

#include <functional>

#include "mlir/IR/Operation.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

using ConsumeOpFn = std::function<void(Operation*)>;
using ConsumeShardingFn = std::function<void(TensorShardingAttr)>;
using TransformShardingFn =
    std::function<TensorShardingAttr(TensorShardingAttr)>;

// Walks the given `rootOp` in forward pre-order and applies `consumeFn` on
// any `TensorShardingAttr` encountered.
//
// In addition, applies `consumeOpFn` on every encountered op, before consuming
// its shardings.
void walkShardings(
    Operation* rootOp, ConsumeShardingFn consumeFn,
    ConsumeOpFn consumeOpFn = [](Operation*) {});

// Walks the given `rootOp` in forward pre-order and replaces any
// `TensorShardingAttr` encountered with the result of applying `transformFn` on
// it.
//
// In addition, applies `consumeOpFn` on every encountered op, before
// transforming its shardings.
void transformShardings(
    Operation* rootOp, TransformShardingFn transformFn,
    ConsumeOpFn consumeOpFn = [](Operation*) {});

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_SHARDING_WALKER_H_
