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

#ifndef SHARDY_DIALECT_SDY_IR_CONSTANTS_H_
#define SHARDY_DIALECT_SDY_IR_CONSTANTS_H_

#include <cstdint>
#include "llvm/ADT/StringRef.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace sdy {

// Tensor sharding attribute name. The attribute will either be of type
// `TensorShardingAttr` for individual tensors (function input/output) or
// `TensorShardingPerValue` for a list of tensor (e.g., the results of an op).
inline constexpr StringRef kShardingAttr = "sdy.sharding";

// Tensor sharding rule attribute name. See OpShardingRuleAttr for more info.
inline constexpr StringRef kShardingRuleAttr = "sdy.sharding_rule";

// Attribute name for saving which input/output/sharding_constraint sharding
// caused a value to be sharded a certain way.
inline constexpr StringRef kShardingOriginsAttr = "sdy.sharding_origins";

// Attribute name for saving which operand/result sharding of an op caused its
// value to be sharded a certain way.
inline constexpr StringRef kPropagationEdgesAttr = "sdy.propagation_edges";

// Attribute name like `kShardingOriginsAttr` but for
// `ShardableDataFlowOpInterface` op block arguments.
inline constexpr StringRef kBlockArgShardingOriginsAttr =
    "sdy.block_arg_sharding_origins";

// Attribute name like `kPropagationEdgesAttr` but for
// `ShardableDataFlowOpInterface` op block arguments.
inline constexpr StringRef kBlockArgPropagationEdgesAttr =
    "sdy.block_arg_propagation_edges";

// Attribute name like `kShardingOriginsAttr` but for
// `ShardableDataFlowOpInterface` op results.
inline constexpr StringRef kResultShardingOriginsAttr =
    "sdy.result_sharding_origins";

// Attribute name like `kPropagationEdgesAttr` but for
// `ShardableDataFlowOpInterface` op results.
inline constexpr StringRef kResultPropagationEdgesAttr =
    "sdy.result_propagation_edges";

// Attribute name for the unique name of a sharding origin. Is either an
// `sdy.sharding_constraint`, or `sdy.ManualComputationOp` input/output.
inline constexpr StringRef kShardingOriginNameAttr = "sdy.sharding_origin_name";

// Default priority for a `DimensionShardingAttr` that doesn't have a
// user-defined priority.
inline constexpr int64_t kDefaultPriority = 0;

// Index of when to use i/j/k as the symbols for factor indices, or to switch
// to z_1/z_2/z_3/....
inline constexpr int kStartAtZ = 'z' - 'i';  // 17

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_CONSTANTS_H_
