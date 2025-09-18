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

#include "shardy/dialect/mpmd/transforms/common/scheduler_preprocess.h"

#include <algorithm>
#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_SCHEDULINGUNITVERIFIERPASS
#define GEN_PASS_DEF_MOVETRANSFERSTOPRODUCERPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

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
