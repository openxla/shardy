/* Copyright 2025 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_IR_ENUMS_H_
#define SHARDY_DIALECT_SDY_IR_ENUMS_H_

namespace mlir {
namespace sdy {

// Represents the type of a factor.
enum class FactorType {
  // The default type, containing the pass-through factors that don't require
  // any communication if sharded in the same way across all tensors that are
  // mapped to them.
  kPassThrough,

  // If we have sharding along reduction dimensions, the partitioner will add
  // all-reduce operations.
  kReduction,

  // If we have sharding along a dimension that needs replication, the
  // partitioner will make this dimension replicated.
  kNeedReplication,

  // If we have sharding along a dimension that needs permutation, the
  // partitioner will add collective-permute operations.
  kPermutation,
};

// Specifies whether the dataflow edge owner sharding is being transformed
// before or after edge propagation.
enum class DataFlowShardingTransformType {
  // Before edge propagation is when the value of the shardings are inspected
  // for propagation.
  kBeforeEdgePropagation,
  // After edge propagation is when the shardings are set back on the data flow
  // edge owner.
  kAfterEdgePropagation
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_ENUMS_H_
