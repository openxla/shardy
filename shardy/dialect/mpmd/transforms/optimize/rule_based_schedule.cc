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
#include <cstddef>

#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_RULEBASEDSCHEDULEPASS
#include "shardy/dialect/mpmd/transforms/optimize/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

class RuleBasedSchedulePass
    : public impl::RuleBasedSchedulePassBase<RuleBasedSchedulePass> {
  using RuleBasedSchedulePassBase::RuleBasedSchedulePassBase;

 private:
  void runOnFunc(FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      return;
    }

    // Build a map from FragmentInfo to FragmentOp for efficient lookup.
    DenseMap<FragmentInfo, FragmentOp, FragmentInfoMapInfo> info_to_op_map;
    func_op.walk([&](FragmentOp fragment) {
      if (IsSchedulingUnit(fragment)) {
        FragmentInfo fragment_info = GetFragmentInfo(fragment);
        if (info_to_op_map.contains(fragment_info)) {
          func_op.emitError()
              << "Fragment info " << llvm::to_string(fragment_info)
              << " already exists for another fragment.";
          return signalPassFailure();
        }
        info_to_op_map[fragment_info] = fragment;
      }
    });

    // A map to keep track of all control dependencies added to fragments.
    // For each fragment, we keep track of number of control-dependencies added.
    // Once we have reordered all the fragments, we can then use this
    // information to remove any control-dependencies from the program.
    DenseMap<FragmentOp, int> ctrl_dependencies;

    // Iterate over the rules and add control dependencies.
    for (const FragmentScheduleRule& rule : rules) {
      if (rule.ordered_fragments.size() < 2) {
        func_op.emitWarning()
            << "Fragment schedule rule must have at least two fragments.";
        continue;
      }

      for (size_t i = 0; i < rule.ordered_fragments.size() - 1; ++i) {
        const FragmentInfo& predecessor_info = rule.ordered_fragments[i];
        const FragmentInfo& successor_info = rule.ordered_fragments[i + 1];

        // Get the fragment operations for the predecessor and successor.
        auto pred_it = info_to_op_map.find(predecessor_info);
        if (pred_it == info_to_op_map.end()) {
          continue;
        }
        FragmentOp predecessor_fragment = pred_it->second;

        auto succ_it = info_to_op_map.find(successor_info);
        if (succ_it == info_to_op_map.end()) {
          continue;
        }
        FragmentOp successor_fragment = succ_it->second;

        if (predecessor_fragment == successor_fragment) {
          continue;
        }
        if (predecessor_fragment.getMeshName() !=
            successor_fragment.getMeshName()) {
          continue;
        }

        // Check if any dataflow dependency already exists.
        if (TargetDependsOnSourceOp(predecessor_fragment, successor_fragment)) {
          continue;
        }

        if (TargetDependsOnSourceOp(successor_fragment, predecessor_fragment)) {
          func_op.emitWarning()
              << "Scheduling rule conflicts with existing dataflow dependency. "
              << "The rule specifies that fragment "
              << llvm::to_string(predecessor_info) << " must run before "
              << llvm::to_string(successor_info)
              << ", but the program already requires the opposite.";
          continue;
        }

        // Add control dependency.
        AddControlDependency(predecessor_fragment, successor_fragment,
                             ctrl_dependencies);
      }
    }

    // Topologically sort the operations to respect the new dependencies. This
    // will also detect any cycles.
    if (!sortTopologically(&func_op.getBody().front())) {
      func_op.emitWarning() << "Cycle detected in the program, not all "
                               "fragments could be properly scheduled.";
    }

    // Remove all control dependencies added.
    RemoveAllControlDependencies(ctrl_dependencies);
  }
};

}  // namespace
}  // namespace mlir::mpmd
