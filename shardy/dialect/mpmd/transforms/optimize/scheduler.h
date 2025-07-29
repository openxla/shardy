/* Copyright 2025 The MPMD Authors.

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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_SCHEDULER_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_SCHEDULER_H_

#include <optional>

#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"

namespace mlir::mpmd {

// Adds scheduling pass. If an `override_must_happen_before` fragment comparator
// is passed in then we ignore any flag-based schedule and use this comparator
// instead.
void AddSchedulingPass(OpPassManager& pm, PipelineSchedule pipeline_schedule,
                       std::optional<FragmentComparator>
                           override_must_happen_before = std::nullopt);

// Adds all passes needed for pipeline scheduling. This includes merge of
// fragments into scheduling units and verification of scheduling units.
//
// When `split_bwd_fragments` is true, then we split backward fragments into
// a fragment whose results are transferred, and one that isn't. This is so that
// we can execute the transfers earlier (e.g. as per Near-Zero Bubble
// Pipeline).
void AddSchedulingPreprocessingPasses(mlir::OpPassManager& pm,
                                      bool split_bwd_fragments,
                                      bool verify_schedule_units);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_SCHEDULER_H_
