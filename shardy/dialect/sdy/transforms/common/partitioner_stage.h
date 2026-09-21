/* Copyright 2026 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PARTITIONER_STAGE_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PARTITIONER_STAGE_H_

#include <cstdint>

namespace mlir {
namespace sdy {

// Controls how far down the Shardy partitioner pipeline to run.
enum class PartitionerStage : uint8_t {
  // No stage requested; defers to `enableInsertExplicitCollectives`.
  kUnspecified = 0,

  // Enables `ShardyResolvePermutationFactorsPass`, which resolves permutation
  // factors via halo exchange.
  kResolvePermutationFactors = 1,

  // Enables `ReshardToCollectivesPass`, which lowers reshards into explicit
  // collective ops.
  kReshardToCollectives = 2,

  // Enables `OptimizeCollectivesPass`, which optimizes the collective ops
  // produced by the previous stage.
  kOptimizeCollectives = 3,

  // Enables `PadForDivisibilityPass`, which pads tensors whose dimensions are
  // not divisible by the axis sizes they are sharded along.
  kPadForDivisibility = 4,

  // Enables `ResolveSingleDeviceShardingPass`, which materializes shardings
  // that live on a single device.
  kResolveSingleDeviceSharding = 5,

  // Enables `ConvertGlobalToLocalPass` followed by `DropShardingAndMeshPass`,
  // which rewrite the module to local shapes and drop the now redundant
  // shardings and meshes.
  kConvertGlobalToLocal = 6,
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PARTITIONER_STAGE_H_
