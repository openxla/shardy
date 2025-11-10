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

#include <utility>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "shardy/dialect/mpmd/transforms/common/merge_fragments.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"
#include "shardy/dialect/mpmd/transforms/optimize/scheduler.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;

void addOptimizePipeline(OpPassManager& pm, OptimizeOptions options) {
  // Adds scheduling and rule-based merging. Rule-based merging must run before
  // other merge passes since those modify the origins of fragments,
  // invalidating the rules.
  AddSchedulingAndRuleBasedMerge(pm, options.pipelineSchedule,
                                 options.fragmentScheduleRules,
                                 options.fragmentMergeRules);

  // The remat passes will run after inlining the call ops and scheduling.
  // The reason why we choose to remat after scheduling is so that we don't need
  // to schedule the remat fragments. For example, given the following fragments
  // on the same mesh in order: F0, F1, B1, B0, if we remat before schedule, we
  // would have F0, F1, F1(remat), B1, F0(remat), B0 and need to worry
  // about scheduling the remat fragments.
  if (options.applyFragmentRemat) {
    pm.addNestedPass<FuncOp>(createRematFragmentPass(
        RematFragmentPassOptions{options.mergeRematFragments}));
  }

  // Verify fragments assigned to the same stage were merged, i.e., it's not
  // possible to have two distinct fragments representing the same stage. Users
  // must use different stages to achieve that kind of program.
  pm.addNestedPass<FuncOp>(createVerifyStageMergingPass());

  if (options.mergeAfterScheduling) {
    AddMergeInferredFragmentsPasses(
        pm, options.absorbInferredFragmentsOnEntryPointFunction,
        options.cloneInferredFragments);
  }

  // Merge fragments as specified by the user.
  if (options.mergeForwardWithBackward) {
    pm.addNestedPass<FuncOp>(createMergeForwardWithBackwardPass());
  }
}

namespace {

struct OptimizePipelineOptions
    : public PassPipelineOptions<OptimizePipelineOptions> {
  Option<bool> mergeAfterScheduling{
      *this, "merge-after-scheduling",
      llvm::cl::desc(
          "Whether to merge inferred fragments only after scheduling."),
      llvm::cl::init(false)};

  Option<PipelineSchedule> pipelineSchedule{
      *this, "pipeline-schedule",
      llvm::cl::desc("The pipeline schedule to use."),
      llvm::cl::init(PipelineSchedule::k1F1B),
      llvm::cl::values(
          clEnumValN(PipelineSchedule::kNone, "none", "No schedule"),
          clEnumValN(PipelineSchedule::k1F1B, "1F1B", "1F1B schedule"),
          clEnumValN(PipelineSchedule::kCircular, "Circular",
                     "Circular schedule"))};
};

}  // namespace

void registerOptimizePipeline() {
  PassPipelineRegistration<OptimizePipelineOptions>(
      "mpmd-optimize-pipeline",
      "Run the standard set of passes to optimize an MPMD program.",
      [](OpPassManager& pm, const OptimizePipelineOptions& pipelineOptions) {
        OptimizeOptions options;
        options.mergeAfterScheduling = pipelineOptions.mergeAfterScheduling;
        options.pipelineSchedule = pipelineOptions.pipelineSchedule;
        addOptimizePipeline(pm, options);
      });
}

}  // namespace mlir::mpmd
