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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_PASSES_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_PASSES_H_

// IWYU pragma: begin_keep

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/distributed_function_pass.h"
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"

// IWYU pragma: end_keep

namespace mlir::mpmd {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/mpmd/transforms/optimize/passes.h.inc"

// Options for the optimize pipeline.
struct OptimizeOptions {
  // A list of fragment merge rules.
  SmallVector<FragmentMergeRule> fragmentMergeRules;
  // A list of fragment schedule rules.
  SmallVector<FragmentScheduleRule> fragmentScheduleRules;
  // Whether to merge inferred fragments only after scheduling.
  bool mergeAfterScheduling = false;
  // Whether to identify matching forward and backward fragments and clone the
  // forward fragment immediately.
  bool applyFragmentRemat = false;
  // Whether remat fragments can be merged with their consumer fragments.
  bool mergeRematFragments = false;
  // Whether to merge forward fragments with backward fragments.
  bool mergeForwardWithBackward = false;
  // Whether to absorb inferred fragments into user-defined fragments on
  // entry-point functions.
  bool absorbInferredFragmentsOnEntryPointFunction = false;
  // Whether to clone inferred fragments when merging.
  bool cloneInferredFragments = false;
  // The pipeline schedule to use.
  PipelineSchedule pipelineSchedule = PipelineSchedule::kGPipe;
};

// Adds the standard set of passes to optimize an MPMD program.
void addOptimizePipeline(OpPassManager& pm, OptimizeOptions options);

// Register the `-mpmd-optimize-pipeline`.
void registerOptimizePipeline();

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_PASSES_H_
