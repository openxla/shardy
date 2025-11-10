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
#include <optional>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
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
        auto [unused_iter, was_inserted] =
            info_to_op_map.insert({fragment_info, fragment});
        if (!was_inserted) {
          func_op.emitError()
              << "Fragment info " << llvm::to_string(fragment_info)
              << " already exists for another fragment. Check if you have "
                 "multiple `named_computation` with the same name. If so, "
                 "please rename one of them.";
          return signalPassFailure();
        }
      }
    });

    // Iterate over the rules and add control dependencies.
    for (const FragmentScheduleRule& rule : rules) {
      if (rule.ordered_fragments.size() < 2) {
        SDY_LOG(WARNING) << "Fragment schedule rule must have at least two "
                            "fragments.\n";
        continue;
      }

      const FragmentInfo* predecessor_info = &rule.ordered_fragments[0];
      for (const FragmentInfo& successor_info :
           llvm::ArrayRef(rule.ordered_fragments).drop_front()) {
        // Ensure predecessor_info is updated to the current successor for the
        // next iteration, unless we encounter an error that should break the
        // dependency chain.
        bool should_update_predecessor_info = true;
        auto predecessor_update_guard = llvm::make_scope_exit([&]() {
          if (should_update_predecessor_info) {
            predecessor_info = &successor_info;
          }
        });

        auto pred_it = info_to_op_map.find(*predecessor_info);
        if (pred_it == info_to_op_map.end()) {
          SDY_LOG(WARNING) << "Fragment " << llvm::to_string(*predecessor_info)
                           << " doesn't exist. Skipping this rule.\n";
          continue;
        }

        auto succ_it = info_to_op_map.find(successor_info);
        if (succ_it == info_to_op_map.end()) {
          // In this case, we don't want to update the predecessor to the
          // successor since the successor doesn't exist.
          should_update_predecessor_info = false;
          SDY_LOG(WARNING) << "Fragment " << llvm::to_string(successor_info)
                           << " doesn't exist. Skipping this rule.\n";
          continue;
        }

        FragmentOp predecessor_fragment = pred_it->second;
        FragmentOp successor_fragment = succ_it->second;
        if (predecessor_fragment == successor_fragment) {
          continue;
        }
        if (predecessor_fragment.getMeshName() !=
            successor_fragment.getMeshName()) {
          SDY_LOG(WARNING) << "Fragment " << llvm::to_string(*predecessor_info)
                           << " and " << llvm::to_string(successor_info)
                           << " are on different meshes. Skipping this rule.\n";
          continue;
        }

        // Check if any dataflow dependency already exists.
        if (GetDependencyPath(predecessor_fragment, successor_fragment)) {
          continue;
        }

        if (std::optional<SmallVector<Operation*>> conflict_path =
                GetDependencyPath(successor_fragment, predecessor_fragment)) {
          SDY_LOG(WARNING) << FormatConflictWarning(
              *predecessor_info, successor_info, *conflict_path);
          continue;
        }

        // For each fragment, we keep track of number of control-dependencies
        // added using an attribute added to the fragment. Once we have
        // reordered all the fragments, we can then use this information to
        // remove any control-dependencies from the program.
        AddControlDependency(predecessor_fragment, successor_fragment);
      }
    }

    // Topologically sort the operations to respect the new dependencies. This
    // will also detect any cycles.
    if (!sortTopologically(&func_op.getBody().front())) {
      SDY_LOG(WARNING) << "Cycle detected in the program, not all "
                          "fragments could be properly scheduled.\n";
    }

    // Remove control dependencies if requested. We would want control
    // dependencies to persist if we expect subsequent passes (like
    // RuleBasedMergePass) to use them during topological sorting.
    if (removeControlDependencies) {
      RemoveAllControlDependencies(func_op);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
