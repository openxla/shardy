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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"
#include "shardy/dialect/mpmd/transforms/optimize/scheduler.h"

namespace mlir::mpmd {

void AddSchedulingAndRuleBasedMerge(
    OpPassManager& pm, PipelineSchedule pipeline_schedule,
    const SmallVector<FragmentScheduleRule>& schedule_rules,
    const SmallVector<FragmentMergeRule>& merge_rules) {
  const bool hasScheduleRules = !schedule_rules.empty();
  const bool hasMergeRules = !merge_rules.empty();

  // Add scheduling: rule-based if rules provided, otherwise pipeline scheduling
  // Remove control dependencies if no merge rules are provided, since there
  // is no later topological sort which will need them.
  if (hasScheduleRules) {
    RuleBasedSchedulePassOptions options;
    options.rules = schedule_rules;
    options.removeControlDependencies = !hasMergeRules;
    pm.addNestedPass<func::FuncOp>(createRuleBasedSchedulePass(options));
  } else {
    AddSchedulingPass(pm, pipeline_schedule,
                      /*removeControlDependencies=*/!hasMergeRules);
  }

  // Add rule-based merge if rules provided
  if (hasMergeRules) {
    RuleBasedMergePassOptions options;
    options.rules = merge_rules;
    options.removeControlDependencies = true;
    pm.addNestedPass<func::FuncOp>(createRuleBasedMergePass(options));
  }
}

}  // namespace mlir::mpmd
