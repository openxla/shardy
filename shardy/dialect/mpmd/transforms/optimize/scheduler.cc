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

#include <algorithm>
#include <cstdint>
#include <optional>

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_PIPELINESCHEDULERPASS
#define GEN_PASS_DEF_SCHEDULINGUNITVERIFIERPASS
#define GEN_PASS_DEF_MOVETRANSFERSTOPRODUCERPASS
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

// Returns the number of microbatches in the program.
// TODO(jupvfranco): This code assumes that microbatching is zero- or one-
// based. Can we generalize this?
uint32_t GetNumMicrobatches(FuncOp func_op) {
  uint32_t max_call_counter = 0;
  bool is_zero_based = false;
  func_op.walk([&max_call_counter, &is_zero_based](FragmentOp fragment) {
    if (auto call_counter = TryToFindCallCounter(fragment)) {
      if (*call_counter == 0) {
        is_zero_based = true;
      }
      max_call_counter = std::max(max_call_counter, *call_counter);
    }
  });
  return max_call_counter + (is_zero_based ? 1 : 0);
}

class SchedulingUnitVerifierPass
    : public impl::SchedulingUnitVerifierPassBase<SchedulingUnitVerifierPass> {
  using SchedulingUnitVerifierPassBase::SchedulingUnitVerifierPassBase;

 private:
  void runOnFunc(FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      return;
    }

    const uint32_t num_microbatches = GetNumMicrobatches(func_op);
    if (num_microbatches == 0) {
      SDY_LOG(WARNING)
          << "Function is not microbatched and therefore cannot be "
             "rescheduled.";
      // We exit instead of emitting an error so that this won't affect init
      // functions that are typically not microbatched.
      return;
    }

    // Check if every mesh has `num_microbatches` scheduling units, half of them
    // forward and the other half backward.
    // TODO(jupvfranco): This works for the simple schedules we support now, but
    // we need to revisit this logic.
    for (NamedMeshAttr mesh : GetSchedulableMeshes(func_op)) {
      int count_fwd = 0, count_bwd = 0;
      for (Operation& op : func_op.getOps()) {
        auto fragment = dyn_cast<FragmentOp>(&op);
        if (!fragment || !IsSchedulingUnit(fragment) ||
            fragment.getMeshName() != mesh.getName()) {
          continue;
        }
        if (*TryToFindSingleTransposeCount(fragment) == 0) {
          count_fwd++;
        } else {
          count_bwd++;
        }
      }
      if (count_fwd != num_microbatches) {
        func_op.emitWarning("Number of forward scheduling units in mesh ")
            << mesh.getName() << " does not match expected number for "
            << num_microbatches << " microbatches. Got " << count_fwd << ".";
      }

      if (count_bwd != num_microbatches) {
        func_op.emitWarning("Number of backward scheduling units in mesh ")
            << mesh.getName() << " does not match expected number for "
            << num_microbatches << " microbatches. Got " << count_bwd << ".";
      }
    }
  }
};

class MoveTransfersToProducerPass
    : public impl::MoveTransfersToProducerPassBase<
          MoveTransfersToProducerPass> {
  using MoveTransfersToProducerPassBase::MoveTransfersToProducerPassBase;

 private:
  void runOnFunc(FuncOp func) override {
    IRRewriter rewriter(func.getContext());
    func.walk([&](TransferOp transfer) {
      if (auto arg = dyn_cast<BlockArgument>(transfer.getOperand())) {
        rewriter.moveOpBefore(transfer, arg.getOwner(),
                              arg.getOwner()->begin());
      } else {
        rewriter.moveOpAfter(transfer, transfer.getOperand().getDefiningOp());
      }
    });
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

void AddSchedulingPreprocessingPasses(OpPassManager& pm,
                                      bool split_bwd_fragments,
                                      bool verify_schedule_units) {
  // The following seems like a good thing to always do, to keep the module
  // more tidy and merged, even if we are not going to actually do any
  // scheduling.
  // Move transfers to right after their producers. Without this pass, if we
  // have a producer fragment followed by transfers, then a consumer fragment,
  // even if the operands of the transfers are from a different producer
  // fragment, we are not able to merge the producer and consumer fragments.
  // This pass moves the transfers to right after the producer, which allows
  // the merge pass to do its job.
  pm.addNestedPass<FuncOp>(createMoveTransfersToProducerPass());
  pm.addNestedPass<FuncOp>(
      createMergeUserDefinedFragmentsIntoSchedulingUnitsPass());
  if (verify_schedule_units) {
    pm.addNestedPass<FuncOp>(createSchedulingUnitVerifierPass());
  }

  // TODO(dvytin): Run split_bwd_fragments independently of the schedule.
  //
  // Furthermore, we now do the split after verification, which ensures that
  // the generic verification code we have still works. But we should consider
  // defining schedule-specific verification conditions (and even passes to
  // prepare the module for a given schedule.)
  // TODO(dvytin): Investigate how to define schedule-specific verification.
  if (split_bwd_fragments) {
    pm.addNestedPass<FuncOp>(createSplitBwdFragmentsPass());
    // TODO(jupvfranco): Do we really need canonicalizations here? Tests seem to
    // fail without it.
    pm.addPass(createCanonicalizerPass(
        GreedyRewriteConfig().setRegionSimplificationLevel(
            GreedySimplifyRegionLevel::Disabled)));
    pm.addNestedPass<FuncOp>(createFragmentDcePass());
  }
}

}  // namespace mlir::mpmd
