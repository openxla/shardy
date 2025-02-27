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

#ifndef SHARDY_ROUND_TRIP_IMPORT_PIPELINES_H_
#define SHARDY_ROUND_TRIP_IMPORT_PIPELINES_H_

#include "mlir/Pass/PassManager.h"

namespace sdy {
namespace round_trip_import {

// Add the sdy-round-trip-import-pipeline in `pm`. The pipeline,
// including a sequence of passes, imports an StableHLO module into the
// SDY (Shardy) dialect.
//
// The module is assumed to have `kShardingRoundTripAttr` and
// `kMeshesRoundTripAttr`.
void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport = true);

// Register the sdy-round-trip-import-pipeline.
void registerSdyRoundTripImportPipeline();

}  // namespace round_trip_import
}  // namespace sdy

#endif  // SHARDY_ROUND_TRIP_IMPORT_PIPELINES_H_
