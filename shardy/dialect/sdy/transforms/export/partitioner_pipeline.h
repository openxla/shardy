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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_PARTITIONER_PIPELINE_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_PARTITIONER_PIPELINE_H_

#include <cstdint>

#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/transforms/common/partitioner_stage.h"

namespace mlir {
namespace sdy {

// Options for addPartitionerPipeline.
struct PartitionerPipelineOptions {
  PartitionerStage stage = PartitionerStage::kConvertGlobalToLocal;
  bool enableHaloExchange = true;
  int64_t replicaCount = 1;
  int64_t partitionCount = 1;
  bool rngBitGeneratorUnsafe = true;
};

// Populates `pm` with the stage-gated partitioner passes up to
// `options.stage`. Expects explicit reshards to already be inserted.
void addPartitionerPipeline(OpPassManager& pm,
                            const PartitionerPipelineOptions& options = {});

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_PARTITIONER_PIPELINE_H_
