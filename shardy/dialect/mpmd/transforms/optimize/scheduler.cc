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

#include "shardy/dialect/mpmd/transforms/optimize/scheduler.h"

#include <optional>

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_PIPELINESCHEDULERPASS
#include "shardy/dialect/mpmd/transforms/optimize/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

// Adds a control dependency in the graph so that `fragment2` depends on
// `fragment1`.
// NOTE: this creates an ill-formed fragment.
void AddControlDependency(FragmentOp fragment1, FragmentOp fragment2,
                          DenseMap<FragmentOp, int>& ctrl_dependency_counter) {
  // We add a new operand at the end.
  int operand_index = fragment2.getNumOperands();
  fragment2->insertOperands(operand_index, {fragment1->getResult(0)});
  ctrl_dependency_counter[fragment2] += 1;
}

// Removes all control dependencies added, so that all fragments are well-formed
// again.
void RemoveAllControlDependencies(
    DenseMap<FragmentOp, int>& ctrl_dependency_counter) {
  for (auto& [fragment, counter] : ctrl_dependency_counter) {
    const int start_index = fragment->getNumOperands() - counter;
    fragment->eraseOperands(start_index, counter);
  }
}

class PipelineSchedulerPass
    : public impl::PipelineSchedulerPassBase<PipelineSchedulerPass> {
  using PipelineSchedulerPassBase::PipelineSchedulerPassBase;

 private:
  void runOnFunc(FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) return;

    // 1. Collect all fragments.
    std::vector<FragmentOp> all_fragments;
    for (Operation& op : func_op.getOps()) {
      if (auto fragment = dyn_cast<FragmentOp>(&op);
          fragment && IsSchedulingUnit(fragment)) {
        all_fragments.push_back(fragment);
      }
    }

    // A map to keep track of all control dependencies added to fragments.
    // For each fragment, we keep track of number of control-dependencies added.
    // Once we have reordered all the fragments, we can then use this
    // information to remove any control-dependencies from the program.
    DenseMap<FragmentOp, int> ctrl_dependencies;

    // 2. Add control dependencies between some pairs of fragments.
    int count_control_dependencies = 0;
    for (FragmentOp fragment1 : all_fragments) {
      for (FragmentOp fragment2 : all_fragments) {
        if (fragment1 == fragment2) continue;

        if (fragment1.getMeshName() != fragment2.getMeshName()) continue;

        // TODO(jupvfranco): we typically expect to see at least 2 * num_stages
        // * num_microbatches. For high numbers of stages or microbatches this
        // may become slow. Consider pre-computing a reachability matrix.
        // Alternatively, consider removing this check completely and define a
        // post-reorder check to verify that for any control-dependency added,
        // the source appears before the destination of the dependency.
        if (TargetDependsOnSourceOp(fragment1, fragment2) ||
            TargetDependsOnSourceOp(fragment2, fragment1)) {
          continue;
        }

        if (mustHappenBefore.value(fragment1, fragment2)) {
          AddControlDependency(fragment1, fragment2, ctrl_dependencies);
          count_control_dependencies++;
        }
      }
    }
    SDY_LOG(INFO) << "Introduced " << count_control_dependencies
                  << " control dependencies for scheduling\n";

    // 3. Sort the graph topologically to guarantee that all dependencies are
    // respected.
    sortTopologically(&func_op.getBody().front());

    // 4. Remove the inserted control-dependencies.
    RemoveAllControlDependencies(ctrl_dependencies);
  }
};

}  // namespace

void AddSchedulingPass(
    OpPassManager& pm, PipelineSchedule pipeline_schedule,
    std::optional<FragmentComparator> override_must_happen_before) {
  PipelineSchedulerPassOptions options;
  options.mustHappenBefore.value = override_must_happen_before.value_or(
      BuiltinFragmentComparator(pipeline_schedule));
  if (!override_must_happen_before) {
    options.mustHappenBefore.schedule = pipeline_schedule;
    SDY_LOG(INFO) << "Reordering computation for "
                  << ToString(pipeline_schedule) << " schedule.";
  }
  pm.addNestedPass<FuncOp>(createPipelineSchedulerPass(options));
}

}  // namespace mlir::mpmd
