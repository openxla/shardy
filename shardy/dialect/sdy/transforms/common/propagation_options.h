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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PROPAGATION_OPTIONS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PROPAGATION_OPTIONS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sdy {

// Options to control the Shardy propagation passes.

struct PropagationOptions {
  // Whether to keep existing and created `sdy::OpShardingRuleAttr` on ops.
  bool keepShardingRules = false;
  // The system directory to dump various rewritten modules for debugging.
  llvm::StringRef dumpDirectory = "";
  // Whether to avoid shardings that may cause values to be non-divisible by its
  // dimension sharding.
  bool conservativePropagation = false;
  // Whether to save debug information about the sharding origins on the module.
  bool debugShardingOrigins = false;
  // Whether to save debug information about the edge shardings on the module.
  bool debugPropagationEdgeSharding = false;
  // Whether to avoid exporting the module for partitioning so that the module
  // will be compatible for another round of propagation.
  bool avoidExportForPartitioning = false;
  // Whether to skip inlining in the module.
  bool skipInline = false;
  // Whether to enable inserting explicit collectives.
  bool enableInsertExplicitCollectives = false;
  // Whether to remove all-gather and reduce-scatter ops for CMV1.
  // TODO(b/432019089): remove this option once CMV1 is completely deprecated.
  bool removeAllGatherReduceScatterForCMV1 = false;
  // Whether automatic partitioning is enabled. If true, an auto-partitioner
  // callback is expected to be registered in `AutoPartitionerRegistry`. The
  // auto-partitioner will be invoked after propagation of user-specified
  // shardings.
  bool enableAutoPartitioning = false;
  // Whether to avoid explicit reshards/collectives on named computations.
  bool avoidReshardsOnNamedComputations = false;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PROPAGATION_OPTIONS_H_
